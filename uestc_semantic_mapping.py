

import torch
import torch.nn.functional as F


def lle_sparse_graph(embeddings, k=10, regularization=1e-3):
    




    embeddings = embeddings.float()
    batch = embeddings.size(0)
    if batch == 1:
        return torch.ones(1, 1, device=embeddings.device, dtype=embeddings.dtype)
    neighbors_count = min(k, batch - 1)
    distances = torch.cdist(embeddings, embeddings)
    distances.fill_diagonal_(float('inf'))
    neighbors = distances.topk(neighbors_count, largest=False).indices
    local = embeddings[neighbors] - embeddings[:, None, :]
    covariance = local @ local.transpose(1, 2)
    trace = covariance.diagonal(dim1=1, dim2=2).sum(dim=1)
    diagonal = (regularization * trace / neighbors_count).clamp_min(1e-4)
    identity = torch.eye(neighbors_count, device=embeddings.device)[None]
    ones = torch.ones(batch, neighbors_count, 1, device=embeddings.device)
    weights = torch.linalg.solve(covariance + diagonal[:, None, None] * identity, ones)
    weights = weights.squeeze(-1)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
    adjacency = embeddings.new_zeros(batch, batch)
    adjacency.scatter_(1, neighbors, weights)
    degree = adjacency.sum(dim=1).clamp_min(1e-8)
    inv_degree = degree.rsqrt()
    return inv_degree[:, None] * adjacency * inv_degree[None, :]


def sgsm_semantic_slope(logits, bert_vectors, k=10, propagate_graph=True):
    








    if logits.size(0) % 2:
        raise ValueError('SGSM expects [original batch; ablated batch]')
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
        batch = logits.size(0) // 2
        semantics = F.softmax(logits.detach().float(), dim=1) @ bert_vectors.detach().float()
        original, ablated = semantics[:batch], semantics[batch:]
        if propagate_graph:
            original = lle_sparse_graph(original, k) @ original
            ablated = lle_sparse_graph(ablated, k) @ ablated
        ratio = (original - ablated) / original.abs().clamp_min(1e-2)
        return torch.nan_to_num(ratio).abs().clamp(max=5)


def sgsm_map(features, slope, spatial_size):
    
    
    
    
    weights = F.interpolate(slope[:, None, :], size=features.size(1),
                            mode='linear', align_corners=False).squeeze(1)
    activation = F.relu((features.float() * weights[:, :, None, None]).sum(dim=1, keepdim=True))
    activation = F.interpolate(activation, size=spatial_size,
                               mode='bilinear', align_corners=False)
    return F.normalize(activation.flatten(1), p=2, dim=1)


def raw_attention_map(features, spatial_size):
    
    activation = features.float().square().mean(dim=1, keepdim=True)
    activation = F.interpolate(activation, size=spatial_size,
                               mode='bilinear', align_corners=False)
    return F.normalize(activation.flatten(1), p=2, dim=1)


def feature_matching_loss(mapped_pairs):
    
    losses = [
        (student - gyro).square().sum(dim=1).mean() +
        (student - acc).square().sum(dim=1).mean()
        for student, gyro, acc in mapped_pairs
    ]
    return sum(losses) / (2 * len(losses))
