function [ y ] = smoothL1( x )
%SMOOTHL1 此处显示有关此函数的摘要
%   此处显示详细说明
    y = sign(x) .* (abs(x)>1) + 2 * x .* (abs(x)<=1);
%     y = sign(x);
%     y = 2 * x;
end

