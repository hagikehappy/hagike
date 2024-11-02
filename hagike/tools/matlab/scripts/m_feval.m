% Matlab万能函数封装器，可处理任意数量的输入与输出
% 若封装的函数的输出为varagout，则假设最高有10个返回值
% 如果函数属于类函数，则 obj 为 varargin 中的第一个元素
% 如果函数无返回值，则返回空数组
% 如果函数有一个返回值，则直接返回值
% 如果函数有多个返回值，则返回列表
% 若指定 numOutputs 为负数（一般为 -1），则自动判断返回值数量
% 否则认为手动指定 numOutputs
function varOutputs = m_feval(func, numOutputs, varargin)
    % func 为函数字符串或函数句柄
    % varargin 为可变数量的参数部分
    if numOutputs < 0
        numOutputs = nargout(func);
    end
    % 若函数使用可变数量输出varagout，假设函数最多返回 10 个值
    maxOutputs = 6;
    if numOutputs == -1
        % 初始化标识符
        varOutputs = repmat({initial_uuid_in_m_feval}, 1, maxOutputs);
        [varOutputs{:}] = feval(func, varargin{:});
        % 确定返回值的实际长度
        numOutputs = 0;
        % 6, 5, 4, 3, 2, 1
        for i = maxOutputs:-1:1
            if ~isequal(varOutputs{i}, initial_uuid_in_m_feval)
                numOutputs = i;
                break;
            end
        end
        % 全空
        if numOutputs == 0
            varOutputs = {};
        % 全满
        elseif numOutputs == maxOutputs
            disp('WARNING: from m_feval, varargout num reaches the boundary!!!');
        % 仅一个
        elseif numOutputs == 1
            varOutputs = varOutputs{1};
        % 中间情况，裁剪 outputs 至合适尺寸
        else
            varOutputs = varOutputs(1:numOutputs);
        end
    % 全空
    elseif numOutputs == 0
        feval(func, varargin{:});
        varOutputs = {};
    % 仅一个
    elseif numOutputs == 1
        varOutputs = feval(func, varargin{:});
    % 中间情况，赋值列表
    else
        varOutputs = cell(1, numOutputs);
        [varOutputs{:}] = feval(func, varargin{:});
    end
end


% 用函数句柄作为初始化标识符
function initial_uuid_in_m_feval()
end
