function w = pivbench_vorticity(u, v)
%PIVBENCH_VORTICITY  ω = ∂v/∂x − ∂u/∂y (sai phân trung tâm, như numpy.gradient).
%   Với toạ độ ảnh (y hướng xuống) dấu ω ngược với hệ trục y hướng lên.
[~, uy] = gradient(u);
[vx, ~] = gradient(v);
w = vx - uy;
end
