The `peak_mem_gb` of the triton_proto rows here is inflated: the harness's edge check
(fixed in 747857c) ran before the peak was read and upcast A to int64. Use
`../26-11-21_14-41-53_sim_triton_proto/` for triton_proto memory. Timings are unaffected,
and the baseline rows (rowsampled, leandense, toolbox) skip the check, so theirs are valid.
