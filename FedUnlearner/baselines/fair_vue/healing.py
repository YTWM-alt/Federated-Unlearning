import torch
from typing import Dict
import math
from FedUnlearner.baselines.fair_vue.subspace import flatten_by_keys  # 顶部import



def selective_kd_heal(student, teacher, dataloader, v_spec,
                      num_steps: int = 60, lr: float = 1e-4,
                      T: float = 2.0, lambda_kd: float = 0.1,
                      lambda_ortho: float = 1e-3, device: str = "cpu",
                      keys=None, grad_project: bool=False,
                      lambda_ce: float = 1.0):   # <== 新增

    import torch, time
    student.train(); teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(student.parameters(), lr=lr)
    ce = torch.nn.CrossEntropyLoss()

    print(f"[HEAL] start healing | steps={num_steps} | lr={lr} | T={T} | "
          f"λ_kd={lambda_kd} | λ_ortho={lambda_ortho} | grad_proj={grad_project}")
    print(f"[HEAL] device={device}, dataloader size={len(dataloader)}")

    # 预处理 v_spec
    Q = None
    if v_spec is not None and v_spec.numel() > 0:
        v_spec = v_spec.to(device).contiguous()
        Q, _ = torch.linalg.qr(v_spec, mode='reduced')  # D×r
        print(f"[HEAL] v_spec loaded, shape={tuple(v_spec.shape)}, rank={Q.shape[1]}")
    else:
        print("[HEAL] no v_spec provided, skip orthogonal penalty")

    t0 = time.time()
    for step, (x, y) in enumerate(dataloader):
        if step >= num_steps: break
        x, y = x.to(device), y.to(device)
        student.zero_grad(set_to_none=True)

        with torch.no_grad():
            tlogits = teacher(x)
        slogits = student(x)

        # 监督 + KD
        loss_task = ce(slogits, y)
        ps = torch.log_softmax(slogits / T, dim=-1)
        pt = torch.softmax(tlogits / T, dim=-1)
        loss_kd = torch.nn.functional.kl_div(ps, pt, reduction="batchmean") * (T ** 2)

        loss = lambda_ce * loss_task + lambda_kd * loss_kd   # <== 改这里

        # 正交惩罚（参数向量按 keys 展平，顺序与 v_spec 对齐）
        loss_ortho = torch.tensor(0., device=device)
        if Q is not None and lambda_ortho > 0:
            if keys is not None:
                vec = flatten_by_keys(student.state_dict(), keys, device=device)
            else:
                vec = torch.cat([p.view(-1) for p in student.parameters() if p.requires_grad])
            coeff = Q.T @ vec
            loss_ortho = coeff.pow(2).sum()
            coef = 0.5 * (1 + math.cos(math.pi * step / max(1, num_steps-1)))  # 退火，从1 -> 0
            loss = loss + (lambda_ortho * coef) * loss_ortho

        # 反向
        loss.backward()

        # （可选）梯度投影到正交补：彻底禁止沿 V_spec 更新
        if grad_project and Q is not None:
            grads, shapes, params = [], [], []
            for p in student.parameters():
                if p.requires_grad and p.grad is not None:
                    grads.append(p.grad.view(-1))
                    shapes.append(p.grad.shape)
                    params.append(p)
            if grads:
                gflat = torch.cat(grads)              # D
                gproj = Q @ (Q.T @ gflat)             # 投影到被遗忘子空间
                gkeep = gflat - gproj                 # 只保留正交补
                # 回填
                offset = 0
                for p, shp in zip(params, shapes):
                    n = math.prod(shp)
                    p.grad.copy_(gkeep[offset:offset+n].view(shp))
                    offset += n

        torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
        opt.step()

        if step % 20 == 0 or step == num_steps - 1:
            print(f"[HEAL step {step:03d}] "
                  f"loss_task={loss_task.item():.4f} | "
                  f"loss_kd={loss_kd.item():.4f} | "
                  f"loss_ortho={loss_ortho.item():.4f} | "
                  f"total={loss.item():.4f}")

    print(f"[HEAL] done in {time.time() - t0:.1f}s\n")