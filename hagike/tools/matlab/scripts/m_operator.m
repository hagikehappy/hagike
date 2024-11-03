% 将算术运算符转为函数，包括：
% 基本算术运算符
%    加法（Addition）：+
%    减法（Subtraction）：-
%    乘法（Multiplication）：*
%    除法（Division）：/
%    左除（Left Division）：\（左除，用于矩阵 Ax = B，x = B \ A）
%    幂运算（Power）：^
% 点运算符
%    点乘（Element-wise Multiplication）：.*
%    点除（Element-wise Division）：./
%    点幂（Element-wise Power）：.^
% 矩阵运算符
%    共轭转置（Conjugate Transpose）：'
%    非共轭转置（Non-conjugate Transpose）：.'
% 逻辑运算符
%    与（AND）：&
%    或（OR）：|
%    非（NOT）：~
%    短路与（SAND）：&&
%    短路或（SOR）：||
%    大于（Bigger）：>
%    大于等于（Bigger Equal）：>=
%    小于（Smaller）: <
%    小于等于（Smaller Equal）: <=
%    等于（Equal）：==
%    不等于（Unequal）：~=
function varOutputs = m_operator(op_type, func, varargin)
    switch(op_type)
        case 'basic'
            varOutputs = basic_operator(func, varargin{:});
        case 'dot'
            varOutputs = dot_operator(func, varargin{:});
        case 'matrix'
            varOutputs = matrix_operator(func, varargin{:});
        case 'logic'
            varOutputs = logic_operator(func, varargin{:});
        otherwise
            error('Unrecognized Operator Type: %s', op_type);
    end
end


function varOutputs = basic_operator(func, varargin)
    switch(func)
        case '+'
            varOutputs = varargin{1} + varargin{2};
        case '-'
            varOutputs = varargin{1} - varargin{2};
        case '*'
            varOutputs = varargin{1} * varargin{2};
        case '/'
            varOutputs = varargin{1} / varargin{2};
        case '\\'
            varOutputs = varargin{1} \ varargin{2};
        case '^'
            varOutputs = varargin{1} ^ varargin{2};
        otherwise
            error('Unrecognized Operator Function: %s in Operator Type Basic', func);
    end
end


function varOutputs = dot_operator(func, varargin)
    switch(func)
        case '.*'
            varOutputs = varargin{1} .* varargin{2};
        case './'
            varOutputs = varargin{1} ./ varargin{2};
        case '.^'
            varOutputs = varargin{1} .^ varargin{2};
        otherwise
            error('Unrecognized Operator Function: %s in Operator Type Dot', func);
    end
end


function varOutputs = matrix_operator(func, varargin)
    switch(func)
        case "'"
            varOutputs = varargin{1}';
        case ".'"
            varOutputs = varargin{1}.';
        otherwise
            error('Unrecognized Operator Function: %s in Operator Type Matrix', func);
    end
end


function varOutputs = logic_operator(func, varargin)
    switch(func)
        case '&'
            varOutputs = varargin{1} & varargin{2};
        case '|'
            varOutputs = varargin{1} | varargin{2};
        case '~'
            varOutputs = ~varargin{1};
        case '&&'
            varOutputs = varargin{1} && varargin{2};
        case '||'
            varOutputs = varargin{1} || varargin{2};
        case '>'
            varOutputs = varargin{1} > varargin{2};
        case '>='
            varOutputs = varargin{1} >= varargin{2};
        case '<'
            varOutputs = varargin{1} < varargin{2};
        case '<='
            varOutputs = varargin{1} <= varargin{2};
        case '=='
            varOutputs = varargin{1} == varargin{2};
        case '~='
            varOutputs = varargin{1} ~= varargin{2};
        otherwise
            error('Unrecognized Operator Function: %s in Operator Type Logic', func);
    end
end

