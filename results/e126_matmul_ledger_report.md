# E126/E127 Matmul ledger

|path|calls|regular|WeightNZ|conversion-like events|
|---|---:|---:|---:|---:|
|nd|1992|1992|0|2344|

## nd call classes
- MLP-labelled calls: 768; projection-labelled calls: 1217
- `decode_cost/lm_head/lm_head`: 7 Matmul calls
- `decode_cost/projection/model.language_model.layers.0.linear_attn.in_proj_qkv`: 5 Matmul calls
- `decode_cost/projection/model.language_model.layers.0.linear_attn.in_proj_z`: 4 Matmul calls
- `decode_cost/projection/model.language_model.layers.0.linear_attn.in_proj_b`: 4 Matmul calls
- `decode_cost/projection/model.language_model.layers.0.linear_attn.in_proj_a`: 4 Matmul calls
- `decode_cost/projection/model.language_model.layers.0.linear_attn.out_proj`: 4 Matmul calls
- `decode_cost/mlp/model.language_model.layers.0.mlp.gate_proj`: 4 Matmul calls
- `decode_cost/mlp/model.language_model.layers.0.mlp.up_proj`: 4 Matmul calls
- `decode_cost/mlp/model.language_model.layers.0.mlp.down_proj`: 4 Matmul calls
- `decode_cost/projection/model.language_model.layers.1.linear_attn.in_proj_qkv`: 4 Matmul calls
- `decode_cost/projection/model.language_model.layers.1.linear_attn.in_proj_z`: 4 Matmul calls
- `decode_cost/projection/model.language_model.layers.1.linear_attn.in_proj_b`: 4 Matmul calls
- `decode_cost/projection/model.language_model.layers.1.linear_attn.in_proj_a`: 4 Matmul calls
- `decode_cost/projection/model.language_model.layers.1.linear_attn.out_proj`: 4 Matmul calls
- `decode_cost/mlp/model.language_model.layers.1.mlp.gate_proj`: 4 Matmul calls
- `decode_cost/mlp/model.language_model.layers.1.mlp.up_proj`: 4 Matmul calls
- `decode_cost/mlp/model.language_model.layers.1.mlp.down_proj`: 4 Matmul calls
- `decode_cost/projection/model.language_model.layers.2.linear_attn.in_proj_qkv`: 4 Matmul calls
- `decode_cost/projection/model.language_model.layers.2.linear_attn.in_proj_z`: 4 Matmul calls
- `decode_cost/projection/model.language_model.layers.2.linear_attn.in_proj_b`: 4 Matmul calls
|nz|1992|1224|768|2344|

## nz call classes
- MLP-labelled calls: 768; projection-labelled calls: 1217
- `decode_cost/lm_head/lm_head`: 7 Matmul calls
- `decode_cost/projection/model.language_model.layers.0.linear_attn.in_proj_qkv`: 5 Matmul calls
- `decode_cost/projection/model.language_model.layers.0.linear_attn.in_proj_z`: 4 Matmul calls
- `decode_cost/projection/model.language_model.layers.0.linear_attn.in_proj_b`: 4 Matmul calls
- `decode_cost/projection/model.language_model.layers.0.linear_attn.in_proj_a`: 4 Matmul calls
- `decode_cost/projection/model.language_model.layers.0.linear_attn.out_proj`: 4 Matmul calls
- `decode_cost/mlp/model.language_model.layers.0.mlp.gate_proj`: 4 Matmul calls
- `decode_cost/mlp/model.language_model.layers.0.mlp.up_proj`: 4 Matmul calls
- `decode_cost/mlp/model.language_model.layers.0.mlp.down_proj`: 4 Matmul calls
- `decode_cost/projection/model.language_model.layers.1.linear_attn.in_proj_qkv`: 4 Matmul calls
- `decode_cost/projection/model.language_model.layers.1.linear_attn.in_proj_z`: 4 Matmul calls
- `decode_cost/projection/model.language_model.layers.1.linear_attn.in_proj_b`: 4 Matmul calls
- `decode_cost/projection/model.language_model.layers.1.linear_attn.in_proj_a`: 4 Matmul calls
- `decode_cost/projection/model.language_model.layers.1.linear_attn.out_proj`: 4 Matmul calls
- `decode_cost/mlp/model.language_model.layers.1.mlp.gate_proj`: 4 Matmul calls
- `decode_cost/mlp/model.language_model.layers.1.mlp.up_proj`: 4 Matmul calls
- `decode_cost/mlp/model.language_model.layers.1.mlp.down_proj`: 4 Matmul calls
- `decode_cost/projection/model.language_model.layers.2.linear_attn.in_proj_qkv`: 4 Matmul calls
- `decode_cost/projection/model.language_model.layers.2.linear_attn.in_proj_z`: 4 Matmul calls
- `decode_cost/projection/model.language_model.layers.2.linear_attn.in_proj_b`: 4 Matmul calls

## E127 conversion summary
- ND transdata-like kernel duration: about 60,834 us; NZ transdata-like classes: about 69,301 us (the paths have different kernel names, so the trace does not prove a new 45 ms cost by name alone).
- The NZ-specific `trans_TransData_5` class totals about 45,099 us; ND has a same-scale `aclnnConvolution_TransDataFzToDst_TransData` class totaling about 44,749 us.
- No `format_cast` or repeated NZ weight-conversion kernel is visible inside decode. The conversion function runs before generation; decoded trace shows reused WeightNZ dispatches.

## Evidence limits
- The trace proves call order, regular versus WeightNZ dispatch, enqueue/dequeue intervals, and matched kernel records.
- M/N/K, dtype, tensor strides, and exact weight identity are unavailable in this profiler export; they remain explicit unknowns rather than inferred values.
- Conversion-like operations are listed from trace names. Their object ownership and reuse lifetime require tensor-level hooks or a profile with input shapes.
- The 1,224 regular calls are attention/projection/lm_head paths outside the current `.mlp.*_proj` conversion selector; shape unsupported is not established.
