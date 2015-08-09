
% MODULE: Prelude. Functional Programming idioms
% Exported functions:
    % typeOf     
    % mapCell  
    % mapRow    
    % mapSlice 
    % zip3Cell   
    % zipCell   
    % zipCellWith
    % foldCell
    % zipColWith
    % zipCol
    
% Derived data structure:
    % Tup      
% Overloaded function:
    % bind      

function M = Prelude()

    b    = @(m,g)  bind(m,g);
    to   = @(m)    typeOf(m);
    mc   = @(g,as) mapCell(g,as);
    mr   = @(g,X)  mapRow(g,X);
    mra  = @(g,X)  mapRowAt(g,X);
    ms   = @(g,X)  mapSlice(g,X);
    t    = @(a,b)  Tup(a,b);
    zc   = @(a,b)  zipCell(a,b);
    zco  = @(a,b)  zipCol(a,b);
    
    z3c  = @(g,a,b,c)  zip3Cell(g,a,b,c);
    zcw  = @(g,a,b)    zipCellWith(g,a,b);
    zcow = @(g,a,b)    zipColWith(g,a,b);
    fc   = @(g,a,b)    foldCell(g,a,b);
    fp   = @(a,b,c)    fixP(a,b,c);
    he   = @(a,b)      hasElem(a,b);

    M = struct(...
        'hasElem'    , he,...
        'bind'       , b,...
        'typeOf'     , to,...
        'mapCell'    , mc,...
        'mapRow'     , mr,...
        'mapRowAt'   , mra,...
        'mapSlice'   , ms,...
        'Tup'        , t,...
        'zip3Cell'   , z3c,...
        'zipCell'    , zc,...
        'zipCellWith', zcw,...
        'zipColWith' , zcow,...
        'zipCol'     , zco,...
        'foldCell'   , fc, ...
        'fixP'       , fp  ...
     );

    M.joinCell  = @(as) joinCell(as);
    M.cellToMat = @(cs) cellToMat(cs);
end


function b = hasElem(a,as)

    b = sum(find(a==as)) > 0;

end


% Find fixed point combinator of some function `g` given init cond `x`
% and termination predicate `pterm`
% fixP :: a -> (a -> a) -> (a -> a -> Double) -> a
function y = fixP(x0,step,pterm)
    
    prv = x0;
    cur = step(x0);
            
    while pterm(prv,cur) == 0
        prv = cur;
        cur = step(prv);
    end
    
    y = cur;

end


function y = foldCell(g,x,xs)

    y = x;
    
    for n = 1:length(xs)
        y = g(xs{n},y);
    end

end


function cs = joinCell(bs)

    cs = {};

    for k = 1:length(bs)

        b = bs{k};

        if isa(b,'cell')
            b_ = joinCell(b);
            cs = {cs{:},b_{:}};
        else
            cs = {cs{:},b};
        end
    end
end

function m = cellToMat(cs)

    m  = [];
    
    if length(cs) ~= 0

        [~,~,p] = size(cs{1});

        if p == 1
            for k = 1:length(cs)
                m(:,:,k) = cs{k};
            end
        end
    end

end


% Ad-hoc oveloading of `bind` to dynamic input types
% bind :: m a -> (a -> m b) -> m b 
function[y] = bind(m,g)

    
    if isType(m,'Maybe')
        y = bindMaybe(m,g);
   else
        y = g(m);
    end
                
end


% bindMaybe :: Maybe a -> (a -> Maybe b) -> Maybe b
function y = bindMaybe(m,g)

    if m.IsNothing == 1
        y = Nothing();
    else 
       y = g(m.Val);
    end
    
end



function b = isType(x1,t2)
    t1 = typeOf(x1);
    l1 = length(t1);
    l2 = length(t2);
    
    if l1 == l2
        if t1 == t2
            b = 1;
        else
            b = 0;
        end
    else
        b = 0;
    end
        
end



% mapCell :: (a -> b) -> {a} -> {b}
function bs = mapCell(g,as)
    for i = 1:length(as)
        as{i} = g(as{i});
    end
    
    bs = as;
end


function Y = mapRow(g,X)

    Y = [];
    for i = 1:size(X,1)
        Y = vertcat(Y,g(X(i,:)));
    end
end


% mapRowAt :: (Int -> a -> a) -> Mat a m n -> Mat a m n
function Y = mapRowAt(g,X)

    Y = [];
    for i = 1:size(X,1)
        Y = vertcat(Y,g(i,X(i,:)));
    end
end


% Map function `g` onto third dimension of 3D matrix `X`.
% Preserve original 3D-ness of matrix
% Mat m n k -> (Mat m n -> Mat m' n') -> Mat m' n' k
function Y = mapSlice(g,X)
    
    [i,j,k] = size(X);
    Y       = g(X(:,:,1));
        
    for x=2:k
        Y(:,:,x) = g(X(:,:,x));
    end
    
end


% Tup :: a -> b -> Struct
function tup = Tup(a,b)
    tup.fst = a;
    tup.snd = b;
end


% typeOf :: a -> String
function t = typeOf(x)
    
    if isa(x,'struct')
        if pField(x,'Type')
            t = x.Type;
        end
    else
        t = class(x);
    end
    
   
end

                        
function ret = pField (x, name)

    if isa(x,'struct') == 1 && length(x) > 0
        ret = go(x,name);
    else
        ret = 0;
    end
    
end


function isFieldResult = go(inStruct, fieldName)
    isFieldResult = 0;
   
    f = fieldnames(inStruct(1));
    for i=1:length(f)
        
      
        if(strcmp(f{i},strtrim(fieldName)))
        isFieldResult = 1;
        return;
    elseif isstruct(inStruct(1).(f{i}))
        isFieldResult = pField(inStruct(1).(f{i}), fieldName);
    if isFieldResult
        return;
    end
    end
    end
end



% zip3Cell :: (a -> b -> c -> d) -> {a} -> {b} -> {c} -> {d}
function rs = zip3Cell(g,as,bs,cs)

    rs = {};
    n  = min(min(length(as),length(bs)),length(cs));
    
    for i = 1:n
        rs{i} = g(as{i},bs{i},cs{i});
    end
end

        
% zipCell :: {a} -> {b} -> {(a,b)}
function cs = zipCell(as,bs)
    cs = zipCellWith(@(a,b) Tup(a,b),as,bs);
end

        


% zipCellWith :: (a -> b -> c) -> {a} -> {b} -> {c}
function cs = zipCellWith(g,as,bs)

    cs = {};
    n  = min(length(as), length(bs));
    
    for i = 1:n
        cs{i} = g(as{i},bs{i});
    end
end



function R = zipColWith(g,M1,M2)
    
    [a,b,c] = size(M1);
    [m,n,k] = size(M2);
    c = min(b,n);
    
    R = [];
    
    for i = 1:c
        
        R = [R,g(M1(:,i),M2(:,i))];
            
    end

end



% Mat m n 1 -> Mat m' n 1 -> Cell (Mat m, Mat m')
function Z = zipCol (X,Y)
    
    m = size(X,2);
    n = size(Y,2);
    Z = {};
    
    for k = 1:m
        Z{k} = Tup(X(:,k),Y(:,k));
     end

end

        