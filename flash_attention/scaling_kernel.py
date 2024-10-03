import mlx.core as mx
from utils.kernel_utils import MetalKernel  

def scaling_spec(scores: mx.array, scaling_factor: float):
    # Direct multiplication without wrapping
    return mx.multiply(scores, scaling_factor)

def scaling_kernel(scores: mx.array, scaling_factor: float):
    header = """
        // No additional headers required for scaling
    """

    source = """
        uint row = thread_position_in_grid.y;
        uint col = thread_position_in_grid.x;

        // Scale each element by the scaling factor
        if (row < scores_shape[0] && col < scores_shape[1]) {
            out[row * scores_shape[1] + col] = scores[row * scores_shape[1] + col] * scaling_factor;
        }
    """

    kernel = MetalKernel(
        name="scaling",
        input_names=["scores", "scaling_factor"],
        output_names=["out"],
        header=header,
        source=source,
    )

    return kernel