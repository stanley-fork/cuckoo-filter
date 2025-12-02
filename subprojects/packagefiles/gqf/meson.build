project('gqf', 'cpp', 'cuda', default_options: ['optimization=3'])

gqf_inc = include_directories('include')

cuda_args = [
    '--expt-extended-lambda',
    '--expt-relaxed-constexpr',
    '--generate-line-info',
    '-gencode=arch=compute_120,code=sm_120',
    '-gencode=arch=compute_75,code=compute_75',
    '-rdc=true',
    '-D QF_BITS_PER_SLOT=16',
    '-D R_BITS=16'
]

gqf_lib = static_library(
    'gqf',
    ['src/gqf.cu', 'src/hashutil.cu', './src/partitioned_counter.cu'],
    include_directories: gqf_inc,
    cuda_args: cuda_args,
)
gqf_dep = declare_dependency(include_directories: gqf_inc, link_with: gqf_lib)