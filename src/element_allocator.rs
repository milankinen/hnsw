use crate::index_params::IndexParams;
use std::mem;

#[repr(C)]
struct Header {
    index: u32,
}

#[repr(C)]
struct Link {}

struct Layer {
    level: u32,
    probability: f64,
    estimated_size_bytes: usize,
    estimated_element_count: usize,
}

impl Layer {
    fn new(params: &IndexParams, level: u32, probability: f64) -> Layer {
        let max_links_per_non_zero_layer = params.m as usize;
        let max_links_per_zero_layer = params.m as usize * 2;
        let max_links_per_element =
            max_links_per_zero_layer + max_links_per_non_zero_layer * level as usize;
        let links_size_in_bytes = mem::size_of::<Link>() * max_links_per_element;
        let data_size_bytes = mem::size_of::<f32>() * params.dimension as usize;
        let bytes_per_element = mem::size_of::<Header>() + links_size_in_bytes + data_size_bytes;
        let estimated_element_count = (probability * params.max_elems as f64) as usize;
        let estimated_size_bytes = estimated_element_count * bytes_per_element;
        Layer {
            level,
            probability,
            estimated_size_bytes,
            estimated_element_count,
        }
    }
}

pub trait MemoryAllocation {
    fn data(&mut self) -> *mut u8;
}

pub struct ElementAllocator {
    params: IndexParams,
    layers: Vec<Layer>,
    memory: Option<Box<dyn MemoryAllocation>>,
}

pub type MemoryAllocator = fn(usize) -> Result<Box<dyn MemoryAllocation>, String>;

pub enum InitializationError {
    MemoryAllocationFailed(String),
}

pub enum ElementAllocationError {
    NotInitialized,
}

struct Element {
    label: u32,
}

impl ElementAllocator {
    pub fn new(params: &IndexParams) -> ElementAllocator {
        let layer_probabilities = calculate_layer_probabilities(&params);
        let layers = layer_probabilities
            .iter()
            .enumerate()
            .map(|(level, p)| Layer::new(&params, level as u32, *p))
            .collect();
        ElementAllocator {
            params: params.clone(),
            layers,
            memory: None,
        }
    }

    pub fn initialize(&mut self, mem_alloc: MemoryAllocator) -> Result<(), InitializationError> {
        match mem_alloc(2) {
            Ok(mem) => {
                self.memory = Some(mem);
                Ok(())
            }
            Err(error) => Err(InitializationError::MemoryAllocationFailed(error)),
        }
    }

    pub fn allocate_element(
        &mut self,
        label: u32,
        data: &[f32],
    ) -> Result<*mut Element, ElementAllocationError> {
        match &mut self.memory {
            Some(mem) => {
                let mut dat = mem.data();
                unsafe {
                    let mut elem = dat.offset(1) as *mut Element;
                    (&mut *elem).label = label;
                    Ok(elem)
                }
            }
            None => Err(ElementAllocationError::NotInitialized),
        }
    }

    fn random_level() -> u32 {
        0
    }
}

fn calculate_layer_probabilities(params: &IndexParams) -> Vec<f64> {
    let m_l = (1. / (params.m as f64)).ln();
    let mut probabilities = vec![];
    let mut level = 0.0;
    loop {
        let p = (-level / m_l).exp() * (1. - (-1. / m_l).exp());
        if p < 1e-12 {
            return probabilities;
        } else {
            probabilities.push(p);
            level += 1.0;
        }
    }
}
