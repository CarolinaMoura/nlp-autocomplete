import torch

def mask_random_positions(tensor, k = 0.15):
    """
    Receives a tensor and returns
    a new tensor of the same shape randomly filled
    with True's at the rate k.
    """
    shape = tensor.shape
    rand = torch.rand(shape)
    return (rand < k)

def mask_end_positions(tensor_ids):
  """
  Receives a tensor and returns
  a tensor of the same shape filled with False's
  and a single True, at the end.
  """
  shape = tensor_ids.shape
  masks = torch.zeros(shape) != 0

  for i in range(0, shape[0]):
    for j in range(0, masks[i].shape[0]):
      if tensor_ids[i][j] == SEP:
        masks[i][j-1] = True
  return masks

  def mask_nothing(tensor):
    """
    Receives a tensor and returns a tensor
    of the same shape filled with False's.
    """
    shape = tensor.shape
    return torch.full(shape, False)