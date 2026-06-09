use crate::atlas::{Atlas, AtlasContent};
use linebender_resource_handle::Blob;
use rect_packer::Rect;
use std::collections::{HashMap, VecDeque};

#[derive(Copy, Clone, Debug)]
pub struct AtlasInfo {
    pub rect: Option<Rect>,
    pub left: i32,
    pub top: i32,
    pub colored: bool,
}

pub enum PixelFormat {
    //TODO: add Rgb(currently we assume Rgba everywhere)
    Rgba,
}

pub struct Image {
    pub width: u32,
    pub height: u32,
    pub data: Blob<u8>,
    pub pixel_format: PixelFormat,
}

/// Rasterized glyph image data (replaces cosmic_text::SwashImage).
pub struct GlyphImage {
    pub data: Blob<u8>,
    pub width: u32,
    pub height: u32,
    pub left: i32,
    pub top: i32,
    /// true = color glyph (goes in color atlas), false = mask glyph (goes in mask atlas)
    pub colored: bool,
}

pub struct GlyphCache {
    pub size: u32,
    pub mask_atlas: Atlas,
    pub color_atlas: Atlas,
    #[allow(clippy::type_complexity)]
    glyph_infos: HashMap<
        (
            u64,      // font blob id
            u16,      // glyph id
            u32,      // font size
            (u8, u8), // subpixel bins (x, y)
            u32,      // synthesis bits (embolden flag + skew)
        ),
        AtlasInfo,
    >,
    svg_infos: HashMap<Vec<u8>, HashMap<(u32, u32), AtlasInfo>>,
    img_infos: HashMap<Vec<u8>, AtlasInfo>,
    /// Image hashes in insertion order, so the oldest same-size region can be
    /// recycled in place when the atlas is full (see `reuse_image_region`).
    img_order: VecDeque<Vec<u8>>,
}

impl GlyphCache {
    pub fn new(device: &wgpu::Device) -> Self {
        let size = 1024;
        Self {
            size,
            mask_atlas: Atlas::new(device, AtlasContent::Mask, size, size),
            color_atlas: Atlas::new(device, AtlasContent::Color, size, size),
            glyph_infos: HashMap::new(),
            img_infos: HashMap::new(),
            svg_infos: HashMap::new(),
            img_order: VecDeque::new(),
        }
    }

    pub fn get_image_mask(&mut self, hash: &[u8], image_fn: impl FnOnce() -> Image) -> AtlasInfo {
        if let Some(info) = self.img_infos.get(hash) {
            return *info;
        }

        let image = image_fn();
        let mut rect = self
            .color_atlas
            .add_region(image.data.data(), image.width, image.height);
        if rect.is_none() {
            // Atlas packer is full. Streaming re-uploads identical-size images
            // every frame, so reuse the oldest cached image region of the same
            // size: overwrite its pixels in place. No growth, no full clear —
            // other images keep their slots, so nothing flickers. (`add_region`
            // set the overflow flag; clear it on success so `check_usage`
            // doesn't recycle the atlas next frame.)
            rect = self.reuse_image_region(image.width, image.height, image.data.data());
            if rect.is_some() {
                self.color_atlas.reset_overflow();
            }
            // If reuse failed (no same-size region — e.g. a new image size), the
            // overflow flag stands and `check_usage` recycles before the next
            // frame's draws, so nothing already painted is corrupted.
        }
        let info = AtlasInfo {
            rect,
            left: 0,
            top: 0,
            colored: true,
        };
        if rect.is_some() {
            self.img_infos.insert(hash.to_vec(), info);
            self.img_order.push_back(hash.to_vec());
        }

        info
    }

    /// Reuse the oldest cached image region whose packed size matches `(w, h)`,
    /// overwriting its pixels with `data` and returning its rect; `None` if no
    /// region of that size exists. Lets same-size streaming frames recycle the
    /// previous generation's regions instead of overflowing the atlas.
    fn reuse_image_region(&mut self, w: u32, h: u32, data: &[u8]) -> Option<Rect> {
        let mut pos = None;
        for (i, hsh) in self.img_order.iter().enumerate() {
            if let Some(r) = self.img_infos.get(hsh).and_then(|info| info.rect) {
                if r.width as u32 == w && r.height as u32 == h {
                    pos = Some(i);
                    break;
                }
            }
        }
        let old_hash = self.img_order.remove(pos?)?;
        let rect = self.img_infos.remove(&old_hash)?.rect?;
        self.color_atlas.overwrite_region(rect, data);
        Some(rect)
    }

    pub fn get_svg_mask(
        &mut self,
        hash: &[u8],
        width: u32,
        height: u32,
        image: impl FnOnce() -> Vec<u8>,
    ) -> AtlasInfo {
        if !self.svg_infos.contains_key(hash) {
            self.svg_infos.insert(hash.to_vec(), HashMap::new());
        }

        {
            let svg_infos = self.svg_infos.get(hash).unwrap();
            if let Some(info) = svg_infos.get(&(width, height)) {
                return *info;
            }
        }

        let data = image();
        let rect = self.color_atlas.add_region(&data, width, height);
        let info = AtlasInfo {
            rect,
            left: 0,
            top: 0,
            colored: true,
        };

        // Don't cache a failed pack (see `get_image_mask`); retry after recycle.
        if rect.is_some() {
            let svg_infos = self.svg_infos.get_mut(hash).unwrap();
            svg_infos.insert((width, height), info);
        }

        info
    }

    /// Look up or rasterize a glyph.
    ///
    /// `synthesis` is an opaque discriminator that differentiates glyphs
    /// rendered with different synthesis settings (e.g. faux bold or italic).
    /// Callers should encode embolden state and skew angle into this value.
    pub fn get_glyph_mask(
        &mut self,
        font_id: u64,
        glyph_id: u16,
        size: u32,
        subpx: (u8, u8),
        synthesis: u32,
        image: impl FnOnce() -> GlyphImage,
    ) -> AtlasInfo {
        let key = (font_id, glyph_id, size, subpx, synthesis);
        if let Some(rect) = self.glyph_infos.get(&key) {
            return *rect;
        }

        let image = image();
        let rect = if image.colored {
            self.color_atlas
                .add_region(image.data.data(), image.width, image.height)
        } else {
            self.mask_atlas
                .add_region(image.data.data(), image.width, image.height)
        };
        let info = AtlasInfo {
            rect,
            left: image.left,
            top: image.top,
            colored: image.colored,
        };
        // Don't cache a failed pack (see `get_image_mask`); retry after recycle.
        if rect.is_some() {
            self.glyph_infos.insert(key, info);
        }
        info
    }

    pub fn update(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        self.mask_atlas.update(device, encoder);
        self.color_atlas.update(device, encoder);
    }

    pub fn check_usage(&mut self, device: &wgpu::Device) -> bool {
        let max_seen = (self.mask_atlas.max_seen as f32 * 2.0)
            .max(self.color_atlas.max_seen as f32 * 2.0) as u32;
        if max_seen > self.size {
            // A region larger than the current atlas appeared (set even when the
            // pack failed): grow to fit. Must take priority over the overflow
            // recycle below, otherwise an image too big for the current atlas
            // would clear-and-retry forever at the same size and never fit.
            self.size = max_seen;
            self.mask_atlas.resize(device, self.size, self.size);
            self.color_atlas.resize(device, self.size, self.size);
            self.clear();
            true
        } else if self.mask_atlas.overflowed() || self.color_atlas.overflowed() {
            // Atlas is big enough but the packer filled up last frame; recycle
            // now — at the start of this frame, before any draws — so the retry
            // packs cleanly and nothing already painted is corrupted. Needed
            // because wide/large images cap area-usage below the 0.7 threshold
            // handled below and so would otherwise never reclaim space.
            self.clear();
            false
        } else if self.mask_atlas.usage() > 0.7 || self.color_atlas.usage() > 0.7 {
            self.clear();
            false
        } else {
            false
        }
    }

    pub fn clear(&mut self) {
        self.mask_atlas.clear();
        self.color_atlas.clear();
        self.glyph_infos.clear();
        self.svg_infos.clear();
        self.img_infos.clear();
        self.img_order.clear();
    }
}
