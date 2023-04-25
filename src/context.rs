use core::mem::size_of;
use core::ops::{Index, IndexMut};

use std::collections::BTreeSet;

use eyre::Result;
use thiserror::Error;

/// The maximum number of dimensions a Tensor can have
const MAX_DIMENSIONS: usize = 4;

/// Initial allocation size for data of a tensor
const INITIAL_DATA_SIZE: usize = 32 * 1024;

/// Initial number of allocations to pre-allocate
const INITIAL_ALLOCATIONS: usize = 5000;

/// The maximum number of graph nodes that can be used
const MAX_GRAPH_NODES: usize = 4096;

/// Read the time stamp using rdtscp to ensure previous instructions have been executed
fn rdtsc() -> u64 {
    let mut x = 0;
    unsafe { std::arch::x86_64::__rdtscp(&mut x) }
}

#[derive(Error, Debug)]
pub enum TensorError {
    /// Out of room for allocating more tensors
    OutOfRoomForTensors,

    /// Ran out of pre-allocated scratch allocations
    OutOfScratchAllocations,

    /// Ran out of pre-allocated leaf nodes in the [`TensorGraph`]
    OutOfLeafNodes,

    /// Ran out of pre-allocated nodes in the [`TensorGraph`]
    OutOfNodes,

    /// Attempted to allocate a tensor with no first dimension
    TensorWithNoFirstDimension,

    /// Incorrect result from building a forward pass
    InvalidForwardGraphComputation,

    /// Attempted to set the value of a tensor of the wrong type
    SetValueError {
        tensor_type: &'static str,
        requested_type: &'static str,
    },

    /// Attempted to compute the given operation without a needed first operand
    FirstOperandNeededForOp {
        tensor: TensorId,
        operation: TensorOp,
    },

    /// Attempted to compute the given operation without a needed second operand
    SecondOperandNeededForOp {
        tensor: TensorId,
        operation: TensorOp,
    },

    Io(std::io::Error),
}

impl core::fmt::Display for TensorError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self:?}")
    }
}

/// The ID used to identify a given tensor
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct TensorId(usize);

macro_rules! impl_index_for_tensor_id {
    ($ty:ty) => {
        impl Index<TensorId> for Vec<$ty> {
            type Output = $ty;
            fn index(&self, index: TensorId) -> &Self::Output {
                &self[index.0]
            }
        }

        impl IndexMut<TensorId> for Vec<$ty> {
            fn index_mut(&mut self, index: TensorId) -> &mut Self::Output {
                &mut self[index.0]
            }
        }

        impl Index<&TensorId> for Vec<$ty> {
            type Output = $ty;
            fn index(&self, index: &TensorId) -> &Self::Output {
                &self[index.0]
            }
        }

        impl IndexMut<&TensorId> for Vec<$ty> {
            fn index_mut(&mut self, index: &TensorId) -> &mut Self::Output {
                &mut self[index.0]
            }
        }
    };
}

impl_index_for_tensor_id!(u8);
impl_index_for_tensor_id!(TensorType);
impl_index_for_tensor_id!(TensorOp);
impl_index_for_tensor_id!([usize; MAX_DIMENSIONS]);
impl_index_for_tensor_id!(TensorId);
impl_index_for_tensor_id!(Option<TensorId>);
impl_index_for_tensor_id!(Vec<u8>);
impl_index_for_tensor_id!(bool);
impl_index_for_tensor_id!(&'static str);
impl_index_for_tensor_id!(TensorData);
impl_index_for_tensor_id!(Option<TensorData>);

/// The type for the given tensor
#[derive(Debug, Copy, Clone)]
pub enum TensorType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q4_2,
    Q4_3,
    Q8_0,
    I8,
    I16,
    I32,
}

#[const_trait]
trait TensorTypeOps {
    fn size(&self) -> usize;
    fn block_size(&self) -> usize;
    fn name(&self) -> &'static str;
}

impl const TensorTypeOps for TensorType {
    fn size(&self) -> usize {
        match self {
            TensorType::F32 => size_of::<f32>(),
            TensorType::I8 => size_of::<i8>(),
            TensorType::I16 => size_of::<i16>(),
            TensorType::I32 => size_of::<i32>(),
            _ => unimplemented!(),
        }
    }
    fn block_size(&self) -> usize {
        match self {
            TensorType::F32
            | TensorType::F16
            | TensorType::I8
            | TensorType::I16
            | TensorType::I32 => 1,
            _ => unimplemented!(),
        }
    }
    fn name(&self) -> &'static str {
        match self {
            TensorType::F32 => "f32",
            TensorType::F16 => "f16",
            TensorType::Q4_0 => "q4_0",
            TensorType::Q4_1 => "q4_1",
            TensorType::Q4_2 => "q4_2",
            TensorType::Q4_3 => "q4_3",
            TensorType::I8 => "i8",
            TensorType::I16 => "i16",
            TensorType::I32 => "i32",
            TensorType::Q8_0 => "q8_0",
        }
    }
}

/// Where to store the result of a [`TensorOp`]
pub enum TensorOpResult {
    /// Store the result in a new tensor
    NewTensor,

    /// Store the result in place of the first operand tensor
    InPlace,
}

#[derive(Debug)]
pub enum TensorData {
    F32(Vec<f32>),
}

impl TensorData {
    pub fn tensor_type(&self) -> TensorType {
        match self {
            TensorData::F32(_) => TensorType::F32,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            TensorData::F32(data) => data.len(),
        }
    }

    pub fn get_f32_mut(&mut self, index: usize) -> &mut f32 {
        match self {
            TensorData::F32(data) => &mut data[index],
            // _ => panic!("Cannot get f32 from non-f32 tensor data"),
        }
    }

    pub fn get_f32(&self, index: usize) -> f32 {
        match self {
            TensorData::F32(data) => data[index],
            // _ => panic!("Cannot get f32 from non-f32 tensor data"),
        }
    }

    pub fn get_f32_slice(&self) -> &[f32] {
        match self {
            TensorData::F32(data) => &data,
            // _ => panic!("Cannot get f32 slice from non-f32 tensor data"),
        }
    }
    pub fn get_f32_slice_mut(&mut self) -> &mut [f32] {
        match self {
            TensorData::F32(data) => data,
            // _ => panic!("Cannot get f32 slice from non-f32 tensor data"),
        }
    }
}

/// A collection of tensors in "Struct of Arrays" memory layout
pub struct Context {
    /// The next tensor ID
    pub next_tensor: usize,

    /// The operation used to compute the data of this tensor
    pub operations: Vec<TensorOp>,

    /// The type of tensor for the tensor at this index
    pub tensor_types: Vec<TensorType>,

    /// The number of dimensions for the tensor at this index
    pub number_of_dimensions: Vec<u8>,

    /// The number of elements in each dimension for this tensor at this index
    pub dimensions: Vec<[usize; MAX_DIMENSIONS]>,

    /// The stride, in bytes, to reach the given dimension in this tensor
    ///
    /// From ggml:
    ///
    /// stride[0] = sizeof(type)
    /// stride[1] = stride[0] * number_of_elements[0] + padding
    /// stride[i] = stride[i - 1] * number_of_elements[i - 1] + padding
    pub strides: Vec<[usize; MAX_DIMENSIONS]>,

    /// The [`TensorId`] holding the gradient for this tensor
    pub gradients: Vec<Option<TensorId>>,

    /// The first operand for the operation for this tensor
    pub first_ops: Vec<Option<TensorId>>,

    /// The second operand for the operation for this tensor
    pub second_ops: Vec<Option<TensorId>>,

    /// The actual data for this tensor
    pub datas: Vec<TensorData>,

    /// Whether this is a root node (name from ggml)
    pub is_root_nodes: Vec<bool>,

    /// Pre-allocated allocations to use in this model
    pub scratch_allocations: Vec<Vec<u8>>,

    /// The name of the type for this tensor
    pub type_names: Vec<&'static str>,

    /// Number of total cycles each tensor operation has taken
    pub operation_stats_cycles: [u64; std::mem::variant_count::<TensorOp>()],

    /// Number of times each tensor operation has occured
    pub operation_stats_count: [u64; std::mem::variant_count::<TensorOp>()],

    /// Total number of cycles monitored so far
    pub operation_stats_total_cycles: u64,
}

impl Context {
    /// Create a new context with an allocated data buffer of the given size
    #[must_use]
    pub fn new(num_init_tensors: usize) -> Context {
        let start = std::time::Instant::now();

        let scratch_allocations = (0..INITIAL_ALLOCATIONS)
            .map(|_| Vec::with_capacity(INITIAL_DATA_SIZE))
            .collect();

        println!("Allocation of scratch took: {:?}", start.elapsed());

        Context {
            next_tensor: 0,
            operations: Vec::with_capacity(num_init_tensors),
            tensor_types: Vec::with_capacity(num_init_tensors),
            number_of_dimensions: Vec::with_capacity(num_init_tensors),
            dimensions: Vec::with_capacity(num_init_tensors),
            strides: Vec::with_capacity(num_init_tensors),
            gradients: Vec::with_capacity(num_init_tensors),
            first_ops: Vec::with_capacity(num_init_tensors),
            second_ops: Vec::with_capacity(num_init_tensors),
            datas: Vec::with_capacity(num_init_tensors),
            is_root_nodes: Vec::with_capacity(num_init_tensors),
            type_names: Vec::with_capacity(num_init_tensors),
            operation_stats_cycles: Default::default(),
            operation_stats_count: Default::default(),
            operation_stats_total_cycles: Default::default(),
            scratch_allocations,
        }
    }

    /// Get one of the pre-allocated allocations to use as data in the model
    pub fn get_allocation(&mut self) -> Result<Vec<u8>> {
        self.scratch_allocations
            .pop()
            .ok_or_else(|| TensorError::OutOfScratchAllocations.into())
    }

    /// Allocate a new tensor ID
    ///
    /// # Panic
    ///
    /// * Out of pre-allocated room for the tensors
    fn new_tensor_id(&mut self) -> TensorId {
        // If we've reached the end of the allocated capacity for tensors..
        if self.next_tensor >= self.operations.capacity() {
            panic!("Out of pre-allocated capacity. Increase number of pre-allocated tensors in Context::new");
        }

        // Get the next tensor ID
        let result = TensorId(self.next_tensor);

        // Increment to the next tensor ID
        self.next_tensor += 1;

        // Return the allocated tensor ID
        result
    }

    /// Check to make sure all of tensor attributes are properly added and are the correct length
    pub fn check_valid_tensors(&self) {
        let Context {
            next_tensor: _,
            scratch_allocations: _,
            operation_stats_cycles: _,
            operation_stats_count: _,
            operation_stats_total_cycles: _,
            operations,
            tensor_types,
            number_of_dimensions,
            dimensions,
            strides,
            gradients,
            first_ops,
            second_ops,
            datas,
            is_root_nodes,
            type_names,
        } = self;

        assert!(operations.len() == operations.len());
        assert!(operations.len() == tensor_types.len());
        assert!(operations.len() == number_of_dimensions.len());
        assert!(operations.len() == dimensions.len());
        assert!(operations.len() == strides.len());
        assert!(operations.len() == gradients.len());
        assert!(operations.len() == first_ops.len());
        assert!(operations.len() == second_ops.len());
        assert!(operations.len() == datas.len());
        assert!(operations.len() == is_root_nodes.len());
        assert!(operations.len() == type_names.len());
    }

    /// Get the number of rows for the given `tensor`
    pub fn number_of_rows(&self, tensor: &TensorId) -> usize {
        // Number of rows is the product of the dimensions for all layers after the first
        self.dimensions(tensor)[1..]
            .iter()
            .filter(|x| **x > 0)
            .product::<usize>()
            .min(1)
    }

    /// Get the [`TensorType`] for the tensor with the given [`TensorId`]
    pub fn tensor_type(&self, id: &TensorId) -> TensorType {
        self.tensor_types[id]
    }

    /// Get the dimensions for the tensor with the given [`TensorId`]
    pub fn dimensions(&self, id: &TensorId) -> [usize; MAX_DIMENSIONS] {
        self.dimensions[id]
    }

    /// Get if the given tensor is a root node
    pub fn is_root_node(&self, id: &TensorId) -> bool {
        self.is_root_nodes[id]
    }

    /// Get the type name for the tensor with the given [`TensorId`]
    pub fn type_name(&self, id: &TensorId) -> &'static str {
        self.type_names[id]
    }

    /// Get the stride for the tensor with the given [`TensorId`]
    pub fn stride(&self, id: &TensorId) -> [usize; MAX_DIMENSIONS] {
        self.strides[id]
    }

    /// Get the number of dimensions for the tensor with the given [`TensorId`]
    pub fn number_of_dimensions(&self, id: &TensorId) -> u8 {
        self.number_of_dimensions[id]
    }

    /// Get the number of dimensions for the tensor with the given [`TensorId`]
    pub fn gradients(&self, id: &TensorId) -> &Option<TensorId> {
        &self.gradients[id]
    }

    /// Set the [`TensorOp`] for the given [`TensorId`]
    pub fn set_tensor_operation(&mut self, id: &TensorId, operation: TensorOp) {
        self.operations[id] = operation;
    }

    /// Set the [`Gradient`] tensor for the given [`TensorId`]
    pub fn set_gradients(&mut self, id: &TensorId, gradient: Option<TensorId>) {
        self.gradients[id] = gradient;
    }

    /// Set the first operand for the given [`TensorId`]
    pub fn set_first_op(&mut self, id: &TensorId, operand: Option<TensorId>) {
        self.first_ops[id] = operand;
    }

    /// Set the second operand for the given [`TensorId`]
    pub fn set_second_op(&mut self, id: &TensorId, operand: Option<TensorId>) {
        self.second_ops[id] = operand;
    }

    /// Get the first operand for the given [`TensorId`]
    pub fn first_op(&self, id: &TensorId) -> Option<TensorId> {
        self.first_ops[id]
    }

    /// Get the second operand for the given [`TensorId`]
    pub fn second_op(&self, id: &TensorId) -> Option<TensorId> {
        self.second_ops[id]
    }

    /// Get the operation for this tensor
    pub fn operation(&self, id: &TensorId) -> TensorOp {
        self.operations[id]
    }

    /// Create a new 1-dimensional tensor of the given `type` with the given number of elements
    pub fn new_tensor_1dim(
        &mut self,
        tensor_type: TensorType,
        num_elements: usize,
    ) -> Result<TensorId> {
        let mut dims = [0; MAX_DIMENSIONS];
        dims[0] = num_elements;
        self._new_tensor_ndim(tensor_type, 1, dims, None)
    }

    /// Create a new 1-dimensional tensor of the given `type` with the given number of elements
    pub fn new_tensor_1dim_data(&mut self, data: TensorData) -> Result<TensorId> {
        let mut dims = [0; MAX_DIMENSIONS];
        dims[0] = data.len();
        let tensor_type = data.tensor_type();
        self._new_tensor_ndim(tensor_type, 1, dims, Some(data))
    }

    /// Create a new n-dimensional (up to [`MAX_DIMENSIONS`]) tensor of the given type and dimensions
    fn _new_tensor_ndim(
        &mut self,
        tensor_type: TensorType,
        tensor_number_of_dimensions: u8,
        tensor_dimensions: [usize; MAX_DIMENSIONS],
        mut data: Option<TensorData>,
    ) -> Result<TensorId> {
        // Allocate the tensor ID for this new tensor
        let new_id = self.new_tensor_id();

        // Get the data allocation for this new tensor
        let data: TensorData = match data.take() {
            Some(data) => data,
            None => {
                // Calculate the number of elements needed by this tensor
                let num_elements = tensor_dimensions
                    .iter()
                    .filter(|x| **x > 0)
                    .product::<usize>()
                    .max(1);

                match tensor_type {
                    TensorType::F32 => TensorData::F32(vec![0.0; num_elements]),
                    _ => unimplemented!(),
                }
            }
        };

        println!("Tensor type: {tensor_type:?} Len: {}", data.len());

        let Context {
            next_tensor: _,
            scratch_allocations: _,
            operation_stats_cycles: _,
            operation_stats_count: _,
            operation_stats_total_cycles: _,
            operations,
            tensor_types,
            number_of_dimensions,
            dimensions,
            strides,
            gradients,
            first_ops,
            second_ops,
            datas,
            is_root_nodes,
            type_names,
        } = self;

        macro_rules! push {
            ($item:ident, $val:expr) => {
                $item
                    .push_within_capacity($val)
                    .map_err(|_| TensorError::OutOfRoomForTensors)?;
            };
        }

        // The first stride is always only the size of the element
        let mut stride = [0; MAX_DIMENSIONS];
        stride[0] = tensor_type.size();

        if tensor_dimensions[0] == 0 {
            return Err(TensorError::TensorWithNoFirstDimension.into());
        }

        // Calculate the stride for the first dimension
        let first_dim = tensor_dimensions[0];
        stride[1] = tensor_type.size() * (first_dim / tensor_type.block_size());

        // Calculate the stride for the remaining dimensions
        for i in 2..(tensor_number_of_dimensions.saturating_sub(1) as usize) {
            let prev_dim = tensor_dimensions[i - 1];
            stride[i] = stride[i - 1] * prev_dim;
        }

        // Allocate a default tensor
        #[rustfmt::skip]
        {
            push!(tensor_types, tensor_type);
            push!(operations, TensorOp::None);
            push!(number_of_dimensions, tensor_number_of_dimensions);
            push!(dimensions, tensor_dimensions);
            push!(gradients, None);
            push!(first_ops, None);
            push!(second_ops, None);
            push!(datas, data);
            push!(strides, stride);
            push!(is_root_nodes, false);
            push!(type_names, tensor_type.name());
        }

        // Check that all of the attributes were added properly
        self.check_valid_tensors();

        // Return the allocated tensor ID
        Ok(new_id)
    }

    /// Alias for [set_param]
    pub fn set_root_node(&mut self, id: &TensorId) -> Result<()> {
        self.set_param(id)
    }

    /// Set that this node is a root node (a parameter)
    pub fn set_param(&mut self, id: &TensorId) -> Result<()> {
        self.is_root_nodes[id] = true;

        let duplicate = self.duplicate(id)?;
        self.gradients[id] = Some(duplicate);

        Ok(())
    }

    /// Alias for [`dup_tensor`]
    pub fn duplicate(&mut self, id: &TensorId) -> Result<TensorId> {
        self.dup_tensor(id)
    }

    /// Duplicate the given tensor and return the new TensorId
    ///
    /// Reference: ggml_dup_tensor
    pub fn dup_tensor(&mut self, id: &TensorId) -> Result<TensorId> {
        let tensor_type = self.tensor_type(id);
        let dimensions = self.dimensions(id);
        let num_dimensions = self.number_of_dimensions(id);
        self._new_tensor_ndim(tensor_type, num_dimensions, dimensions, None)
    }

    /// Duplicate the given tensor and return the new TensorId
    ///
    /// Reference: ggml_view_tensor
    pub fn view_tensor(&mut self, id: &TensorId) -> Result<TensorId> {
        let tensor_type = self.tensor_type(id);
        let dimensions = self.dimensions(id);
        let num_dimensions = self.number_of_dimensions(id);
        let _result = self._new_tensor_ndim(tensor_type, num_dimensions, dimensions, None);
        panic!();
    }

    /// Multiply the given operands `op1` and `op2` and return the resulting tensor
    pub fn mul(&mut self, op1: &TensorId, op2: &TensorId) -> Result<TensorId> {
        self._tensor_operation(TensorOp::Mul, TensorOpResult::NewTensor, op1, Some(op2))
    }

    /// Add the given operands `op1` and `op2` and return the resulting tensor
    pub fn add(&mut self, op1: &TensorId, op2: &TensorId) -> Result<TensorId> {
        self._tensor_operation(TensorOp::Add, TensorOpResult::NewTensor, op1, Some(op2))
    }

    /// Calulate the ReLU for each value in the given tensor
    pub fn relu(&mut self, op1: &TensorId) -> Result<TensorId> {
        self._tensor_operation(TensorOp::Relu, TensorOpResult::NewTensor, op1, None)
    }

    /// Calculate the soft max of the values in the given tensor
    pub fn soft_max(&mut self, op1: &TensorId) -> Result<TensorId> {
        self._tensor_operation(TensorOp::SoftMax, TensorOpResult::NewTensor, op1, None)
    }

    /// Create the given tensor operation using the operands for this tensor
    pub fn _tensor_operation(
        &mut self,
        operation: TensorOp,
        dest: TensorOpResult,
        op1: &TensorId,
        op2: Option<&TensorId>,
    ) -> Result<TensorId> {
        // Default to this operation not needing a gradient tensor
        let mut is_node = false;

        // If the result is a new tensor or either of the operands already has a gradient, then this
        // is a node and will create a gradient tensor
        let op1_has_grad = self.gradients(op1).is_some();
        let op2_has_grad = op2.and_then(|op| self.gradients(op).as_ref()).is_some();
        if matches!(dest, TensorOpResult::NewTensor) && (op1_has_grad || op2_has_grad) {
            is_node = true;
        }

        // Allocate a new tensor based on the destination of this operation
        let result = match dest {
            TensorOpResult::InPlace => self.view_tensor(op1)?,
            TensorOpResult::NewTensor => self.duplicate(op1)?,
        };

        let gradient = if is_node {
            Some(self.duplicate(&result)?)
        } else {
            None
        };

        self.set_tensor_operation(&result, operation);
        self.set_gradients(&result, gradient);
        self.set_first_op(&result, Some(op1.clone()));
        self.set_second_op(&result, op2.and_then(|op| Some(op.clone())));

        println!(
            "{result:?} {operation:?} {:?} {:?}",
            self.first_op(&result),
            self.second_op(&result)
        );

        // Return the resulting tensor
        Ok(result)
    }

    pub fn build_forward(&self, tensor: &TensorId) -> Result<TensorGraph> {
        const EMPTY_TENSOR_ID: Option<TensorId> = None;

        let mut graph = TensorGraph {
            visited: BTreeSet::new(),
            next_node_index: 0,
            next_leaf_index: 0,
            nodes: [EMPTY_TENSOR_ID; MAX_GRAPH_NODES],
            leafs: [EMPTY_TENSOR_ID; MAX_GRAPH_NODES],
            gradients: [EMPTY_TENSOR_ID; MAX_GRAPH_NODES],
        };

        // Build the forward pass from this tensor
        self._build_forward(&mut graph, tensor)?;

        // If correctely computed, the last node in the graph is the given `tensor`.
        match &graph.nodes[graph.next_node_index - 1] {
            None => Err(TensorError::InvalidForwardGraphComputation.into()),
            Some(last_tensor) if last_tensor != tensor => {
                Err(TensorError::InvalidForwardGraphComputation.into())
            }
            _ => Ok(graph),
        }
    }

    /// Build the forward compute graph into the given [`TensorGraph`] ending in the given `tensor`
    ///
    /// Reference: ggml_build_forward_impl
    pub fn _build_forward(&self, graph: &mut TensorGraph, tensor: &TensorId) -> Result<()> {
        // Insert the tensor to the visited set and early exit if we have seen this tensor before
        if !graph.visited.insert(tensor.clone()) {
            println!("Tensor: {tensor:?} already seen.. skipping..");
            return Ok(());
        }

        let operation = &self.operations[tensor];

        println!("Tensor: {tensor:?} Operation: {operation:?}");

        if let Some(first_op) = self.first_op(tensor) {
            self._build_forward(graph, &first_op)?;
        }

        if let Some(second_op) = self.second_op(tensor) {
            self._build_forward(graph, &second_op)?;
        }

        let curr_operation = self.operation(tensor);
        let is_leaf_node =
            matches!(curr_operation, TensorOp::None) && self.gradients(tensor).is_none();

        if is_leaf_node {
            // Ensure that there are enough leaf nodes still in the graph
            if graph.next_leaf_index >= MAX_GRAPH_NODES {
                return Err(TensorError::OutOfLeafNodes.into());
            }

            // Add leaf node to the graph
            graph.leafs[graph.next_leaf_index] = Some(tensor.clone());
            graph.next_leaf_index += 1;
        } else {
            // Ensure that there are enough leaf nodes still in the graph
            if graph.next_node_index >= MAX_GRAPH_NODES {
                return Err(TensorError::OutOfNodes.into());
            }

            // Add node and its gradient tensor to the graph
            graph.nodes[graph.next_node_index] = Some(tensor.clone());
            graph.gradients[graph.next_node_index] = self.gradients(tensor).clone();
            graph.next_node_index += 1;
        }

        Ok(())
    }

    /// Set the given `tensor` to the given `value`
    pub fn set_value_f32(&mut self, tensor: &TensorId, value: f32) -> Result<()> {
        let TensorData::F32(data) = &mut self.datas[tensor];
        /*
        else {
            return Err(TensorError::SetValueError {
                tensor_type: self.type_name(tensor),
                requested_type: "f32"
            }.into());
        };
        */

        // Set the values of this data to the given value
        data.iter_mut().for_each(|x| *x = value);

        // Reset the data buffer
        // self.datas[tensor] = Some(TensorData::F32(data));

        Ok(())
    }

    /// Dump the [`TensorGraph`] as a .dot graph
    pub fn dump_dot(&mut self, graph: &mut TensorGraph) {
        let mut result = String::new();
        result.push_str("digraph {");
        result.push_str(" newrank = true;\n");
        result.push_str(" rankdir = LR;\n");
        let mut nodes = graph.nodes.iter();
        let mut leafs = graph.leafs.iter();

        // Add all of the nodes as nodes in the dot graph
        while let Some(Some(node)) = nodes.next() {
            let mut color = "white";
            if self.is_root_node(node) {
                color = "lightblue";
            }

            let op = self.operation(node);
            result.push_str(&format!("\"{node:?}\" [ style = filled; fillcolor = {color}; shape = record; label = \"{op:?}\"; ]\n"));
        }

        // Add all of the leaf nodes as nodes in the dot graph
        while let Some(Some(node)) = leafs.next() {
            let mut color = "white";
            if self.is_root_node(node) {
                color = "blue";
            }

            let op = self.operation(node);
            result.push_str(&format!("\"{node:?}\" [ style = filled; fillcolor = {color}; shape = record; label = \"{op:?}\"; ]\n"));
        }

        for node_id in 0..graph.next_node_index {
            let Some(node) = &graph.nodes[node_id] else { break; };

            if let Some(first_op) = self.first_op(node) {
                let first_op_parent = self.get_graph_parent(graph, &first_op);
                let node_parent = self.get_graph_parent(graph, node);

                let src = match first_op_parent {
                    Some(parent) => parent,
                    None => first_op,
                };

                let dst = match node_parent {
                    Some(ref parent) => parent,
                    None => node,
                };

                result.push_str(&format!("\"{src:?}\" -> \"{dst:?}\";\n"));
            }

            if let Some(second_op) = self.second_op(node) {
                let second_op_parent = self.get_graph_parent(graph, &second_op);
                let node_parent = self.get_graph_parent(graph, node);

                let src = match second_op_parent {
                    Some(parent) => parent,
                    None => second_op,
                };

                let dst = match node_parent {
                    Some(ref parent) => parent,
                    None => node,
                };

                result.push_str(&format!("\"{src:?}\" -> \"{dst:?}\";\n"));
            }
        }

        for node_id in 0..graph.next_leaf_index {
            let Some(leaf) = graph.leafs[node_id] else { break };

            if let Some(first_op) = self.first_op(&leaf) {
                result.push_str(&format!(
                    "\"{first_op:?}\" -> \"{leaf:?}\" [ label = 'leaf' ];\n"
                ));
            }

            if let Some(second_op) = self.second_op(&leaf) {
                result.push_str(&format!(
                    "\"{second_op:?}\" -> \"{leaf:?}\" [ label = 'leaf' ];\n"
                ));
            }
        }

        result.push_str("}");

        println!("{result}");

        std::fs::write("/tmp/graph.dot", result).unwrap();
    }

    pub fn print_tensor(&self, tensor: &TensorId) {
        let dims = self.dimensions(&tensor);
        let op = self.operation(&tensor);
        let node_type = if self.is_root_node(&tensor) {
            "Root"
        } else if self.gradients(&tensor).is_some() {
            "Grad"
        } else {
            "NA"
        };

        log::debug!("{tensor:?} | {dims:?} {op:?} {node_type}");
    }

    /// Print the graph in ASCII form
    pub fn print_graph(&self, graph: &TensorGraph) {
        for i in 0..graph.next_node_index {
            let Some(node) = graph.nodes[i] else {
                println!("{i} is not a node?");
                continue;
            };

            let dims = self.dimensions(&node);
            let op = self.operation(&node);
            let node_type = if self.is_root_node(&node) {
                "Root"
            } else if self.gradients(&node).is_some() {
                "Grad"
            } else {
                "NA"
            };

            println!("Node {i}: {node:?} | {dims:?} {op:?} {node_type}");
        }

        for i in 0..graph.next_leaf_index {
            let Some(node) = graph.leafs[i] else {
                println!("{i} is not a leaf?");
                continue;
            };

            let dims = self.dimensions(&node);
            let op = self.operation(&node);

            println!("Leaf {i}: {node:?} | {dims:?} {op:?}");
        }
    }

    ///
    pub fn add_stat(&mut self, op: TensorOp, new_cycles: u64) {
        // Add the new elapsed cycles to this operation
        self.operation_stats_cycles[op as usize] += new_cycles;

        // Increment the count for this operation
        self.operation_stats_count[op as usize] += 1;

        // Add the new elapsed cycles to the total number of cycles executed
        self.operation_stats_total_cycles += new_cycles;
    }

    /// Compute the forward pass for each node's operation
    pub fn compute_forward(&mut self, graph: &TensorGraph) -> Result<()> {
        let mut nodes = graph.nodes.iter();
        while let Some(Some(node)) = nodes.next() {
            // Get the operation for the current node
            let op = self.operation(node);
            log::debug!("Compute node: {node:?} {op:?}");
            self.print_tensor(node);

            if matches!(op, TensorOp::None) {
                continue;
            }

            let start = rdtsc();

            match op {
                TensorOp::Mul => self.compute_forward_mul(node)?,
                TensorOp::Add => self.compute_forward_add(node)?,
                TensorOp::Relu => self.compute_forward_relu(node)?,
                TensorOp::SoftMax => self.compute_forward_softmax(node)?,
                _ => unimplemented!("{op:?}"),
            }

            let elapsed = rdtsc() - start;
            self.add_stat(op, elapsed);

            // Update the statistics for this operation
        }

        Ok(())
    }

    /// Get the parent node for the given node in the given graph
    pub fn get_graph_parent(&self, graph: &TensorGraph, node: &TensorId) -> Option<TensorId> {
        for node_id in 0..graph.next_node_index {
            let Some(parent) = graph.nodes[node_id] else {  break  };

            if self.gradients(&parent) == &Some(*node) {
                return Some(parent);
            }
        }

        None
    }

    pub fn print_stats(&self) {
        for (i, val) in self.operation_stats_cycles.iter().enumerate() {
            println!(
                "{i} {:?}: {val} count {} ({} each iter)/ {}",
                TensorOp::from(i),
                self.operation_stats_count[i],
                val / self.operation_stats_count[i].max(1),
                self.operation_stats_total_cycles
            );
        }
    }
}

#[derive(Debug, Default, Copy, Clone)]
pub enum TensorOp {
    #[default]
    None,
    Mul,
    Add,
    Relu,
    SoftMax,
}

impl From<usize> for TensorOp {
    fn from(val: usize) -> TensorOp {
        match val {
            0 => TensorOp::None,
            1 => TensorOp::Mul,
            2 => TensorOp::Add,
            3 => TensorOp::Relu,
            4 => TensorOp::SoftMax,
            _ => unimplemented!("TensorOp::from({val})"),
        }
    }
}

pub struct TensorGraph {
    visited: BTreeSet<TensorId>,
    next_node_index: usize,
    next_leaf_index: usize,
    nodes: [Option<TensorId>; MAX_GRAPH_NODES],
    gradients: [Option<TensorId>; MAX_GRAPH_NODES],
    leafs: [Option<TensorId>; MAX_GRAPH_NODES],
}
