function S = pivbench_synth(kind, fieldParams, shape, imgParams)
%PIVBENCH_SYNTH  Sinh cặp ảnh hạt tổng hợp kèm trường gốc (dùng bộ sinh Python của pivbench).
%   S = pivbench_synth('wake', struct('D',140,'U_inf',6,'x_base',30), [256 384], struct('ppp',0.05,'snr',10))
%   S = pivbench_synth('sine', struct('amplitude',0.5,'wavelength',32), [256 256])
%   S = pivbench_synth('lamb_oseen', struct('core_radius',16,'u_max',4), [256 256])
%
%   kind       : 'wake' | 'sine' | 'lamb_oseen' | 'turbulence' | 'uniform'
%   fieldParams: tham số trường (xem src/pivbench/synth/fields.py, wake.py — WakeParams)
%   imgParams  : ppp, d_p, snr, sheet_thickness, out_of_plane_std, seed, bit_depth, ...
%   S          : struct img1, img2 (double), u_true, v_true, valid (logical, false = thân mô hình), meta
if nargin < 2, fieldParams = struct(); end
if nargin < 3 || isempty(shape), shape = [256 256]; end
if nargin < 4, imgParams = struct(); end
job.action = 'synth';
job.shape = double(shape);
job.field_json = jsonencode(struct('kind', kind, 'params', fieldParams));
if isempty(fieldnames(imgParams))
    job.params_json = '';
else
    job.params_json = jsonencode(imgParams);
end
res = pivbench_call(job);
S.img1 = res.img1;
S.img2 = res.img2;
S.u_true = res.u_true;
S.v_true = res.v_true;
S.valid = logical(res.valid);
S.meta = jsondecode(res.meta_json);
end
