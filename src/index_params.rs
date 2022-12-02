pub struct IndexParams {
    pub dimension: u32,
    pub max_elems: u32,
    pub m: u32,
    pub m0: u32,
}

impl Clone for IndexParams {
    fn clone(&self) -> Self {
        IndexParams { ..*self }
    }
}