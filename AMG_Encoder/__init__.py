import torch
from torch.amp import custom_fwd, custom_bwd
from ._C import createTransformationMatricesForward, \
        createTransformationMatricesBackward, \
        encodeForward, encodeBackward, \
        featureDensityForward, featureDensityBackward, \
        encodeForwardTransform, encodeBackwardTransform, \
        featureDensityForwardTransform, featureDensityBackwardTransform

class CreateTransformationMatricesFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type="cuda")
    def forward(ctx, rotations, scales, translations):
        ctx.save_for_backward(rotations, scales, translations)

        result = createTransformationMatricesForward(rotations, scales, translations)

        return result

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        rotations, scales, translations = ctx.saved_tensors

        grad_rotations, grad_scales, grad_translations = createTransformationMatricesBackward(rotations, scales, translations, grad_output)

        return grad_rotations, grad_scales, grad_translations

class EncodeCoordinates(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type="cuda", cast_inputs=torch.float16)
    def forward(ctx, query_coordinates : torch.Tensor, rotations : torch.Tensor, 
                scales : torch.Tensor, translations : torch.Tensor, feature_grids : torch.Tensor):
        permuted_query = query_coordinates.permute(1, 0).contiguous()
        
        feature_vectors = encodeForward(permuted_query, rotations, 
                                        scales, translations, feature_grids)
        if any(ctx.needs_input_grad[1:]):  # Check if any input requires gradient
            ctx.save_for_backward(permuted_query, rotations, 
                        scales, translations, feature_grids)
        return feature_vectors

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        if not ctx.saved_tensors:
            return None, None, None, None, None
        
        query_coordinates, rotations, scales, \
            translations, feature_grids = ctx.saved_tensors

        grad_feature_grids = encodeBackward(query_coordinates, 
            rotations, scales, translations, feature_grids, grad_output)
        
        return None, None, None, None, grad_feature_grids
    
class EncodeCoordinates_TransformationMatrix(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type="cuda", cast_inputs=torch.float16)
    def forward(ctx, query_coordinates : torch.Tensor, transformation_matrices : torch.Tensor, feature_grids : torch.Tensor):
        permuted_query = query_coordinates.permute(1, 0).contiguous()
        
        feature_vectors = encodeForwardTransform(permuted_query, transformation_matrices, feature_grids)
        if any(ctx.needs_input_grad[1:]):  # Check if any input requires gradient
            ctx.save_for_backward(permuted_query, transformation_matrices, feature_grids)
        return feature_vectors

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        if not ctx.saved_tensors:
            return None, None, None
        
        query_coordinates, transformation_matrices, feature_grids = ctx.saved_tensors

        grad_feature_grids = encodeBackwardTransform(query_coordinates, 
            transformation_matrices, feature_grids, grad_output)
        
        return None, None, grad_feature_grids

class FeatureDensity(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query_coordinates, rotations, scales, translations):
        density = featureDensityForward(query_coordinates, rotations, 
                                           scales, translations)

        # Store for use in backward
        ctx.save_for_backward(query_coordinates, rotations, 
                        scales, translations)

        return density

    @staticmethod
    def backward(ctx, grad_output):
        query_coordinates, rotations, \
            scales, translations = ctx.saved_tensors
        
        dL_dRotations, dL_dScales, dL_dTranslations = featureDensityBackward(query_coordinates, 
            rotations, scales, translations, grad_output)

        return None, dL_dRotations, dL_dScales, dL_dTranslations

class FeatureDensity_TransformationMatrix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query_coordinates, transformation_matrices):
        density = featureDensityForwardTransform(query_coordinates, transformation_matrices)

        # Store for use in backward
        ctx.save_for_backward(query_coordinates, transformation_matrices)

        return density

    @staticmethod
    def backward(ctx, grad_output):
        query_coordinates, transformation_matrices = ctx.saved_tensors
        
        dL_dTransformationMatrices = featureDensityBackwardTransform(query_coordinates, 
            transformation_matrices, grad_output)

        return None, dL_dTransformationMatrices

def create_transformation_matrices(rotations, scales, translations) -> torch.Tensor:
    return CreateTransformationMatricesFunction.apply(rotations, scales, translations)

def encode(query_coordinates, rotations, scales, translations, feature_grids) -> torch.Tensor:
    return EncodeCoordinates.apply(query_coordinates, 
        rotations, scales, translations, feature_grids)


def feature_density(query_coordinates, rotations, scales, translations) -> torch.Tensor:
    return FeatureDensity.apply(query_coordinates, 
        rotations, scales, translations)

def encode_transformation_matrix(query_coordinates, transformation_matrices, feature_grids) -> torch.Tensor:
    return EncodeCoordinates_TransformationMatrix.apply(query_coordinates, 
        transformation_matrices, feature_grids)


def feature_density_transformation_matrix(query_coordinates, transformation_matrices) -> torch.Tensor:
    return FeatureDensity_TransformationMatrix.apply(query_coordinates, 
        transformation_matrices)