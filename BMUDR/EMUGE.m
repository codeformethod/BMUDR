function [F,obj,sumTime,w]=BMUDR(X,m,projectionDim,gamma,alpha,beta)
% objection: reduce the dimensionality of original dataset from d to d'
% X_v: d_v*n, original v-th view dataset
% U_v: d_v*m, anchor set of v-th view
% Z: m*n, anchor-data similarity matrix
% n: data scale
% m: anchor number
% n_v: view number
% w_v: n_v*1, weights for v-th view
% F: n*d' low-dimensional representation
% Written by Qianyao Qiang (qianyao.qiang AT polyu.edu.com)

tic
n_v=size(X,2);
n=size(X{1,1},1);
NITER=100;
thresh=1e-5;
[n,~] = size(X{1,1});

%% Randomly initialization
for v=1:n_v
    w(v)=1/n_v;
end
S=rand(n,n);
col_sum = sum(S,2);
S = S./col_sum;
Z=ones(m,n)/m;

options = optimset( 'Algorithm','interior-point-convex','Display','off');
%% updating
for iter = 1:NITER
    %% 1. Updating anchor set U_v
    U={};
    ZZ=Z*Z';
    invZZ=pinv(ZZ);%ÇóÄæ²Ù×÷ÔÙÕå×Ã
    for v=1:n_v
        U{1,v} = X{1,v}'*Z'*invZZ;
    end
    
    %% 2. Updating low-dimensional representation E
    [~,F,G,~,~,~,]=svd2uv(Z',projectionDim);
    E=[F;G];
    
    %% 3. Updating anchor-data similarity matrix Z
    tmp=(w(1)^gamma)*(U{1,1})'*(U{1,1});
    for v=2:n_v
        tmp=tmp+(w(v)^gamma)*(U{v})'*(U{v});
    end
    H=2*(tmp+alpha*eye(m));
    H=(H+H')/2;
     
    DF=(sum(Z)).^(-1/2);
    DG=(sum(Z')).^(-1/2);
    D_FG=zeros(n,m);
    for i=1:n
        for j=1:m
            D_FG(i,j)=(norm(((F(i,:)*DF(i)-F(j,:)*DG(j))),'fro'))^2;
        end
    end
    tmp1=zeros(1,m);
    for i=1:n
        tmp1=(w(1)^gamma)*(X{1,1}(i,:))*U{1,1};
        for v=2:n_v
            tmp1=tmp1+(w(v)^gamma)*(X{1,v}(i,:))*U{1,v};
        end
        ff=beta*D_FG(i,:)-2*tmp1;
        Z(:,i)=quadprog(H,ff',[],[],ones(1,m),1,zeros(m,1),ones(m,1),Z(:,i),options);
    end
    
    %% Updating weight coefficient vector w
    hv=zeros(n_v,1);
    for v=1:n_v
        hv(v)=(norm((X{1,v}'-U{1,v}*Z),'fro')^2)^(1/(1-gamma));
    end
    for v=1:n_v
       w(v)=hv(v)/sum(hv);
    end
    
    S=(S+S')/2;
    Ls=(diag(sum(S))-S);
    
    sum1=0;
    for v=1:n_v
        sum1=sum1+(w(v)^gamma)*(norm((X{1,v}'-U{1,v}*Z),'fro')^2);
    end
    sum2=norm(Z,'fro')^2;
    ver=version;
    if(str2double(ver(1:3))>=9.1)
        Z = Z./sum(Z,1);% for the newest version(>=9.1) of MATLAB
    else
        Z = bsxfun(@rdivide, Z, sum(Z,1));% for old version(<9.1) of MATLAB
    end
    z1 = sum(Z,1);
    D1z = spdiags(1./sqrt(z1'),0,n,n);  
    z2 = sum(Z,2);
    D2z = spdiags(1./sqrt(z2),0,m,m);
    Z1 = D2z*Z*D1z;
    O_3 = eye(projectionDim)-2*F'*Z1'*G;
    sum3=trace(O_3);
    obj(iter,1)=sum1+alpha*sum2+beta*sum3;
    
    if iter>2 && abs((obj(iter)-obj(iter-1))/obj(iter))<thresh
        break;
    end
    if iter>30 && sum(abs(obj(iter-9:iter-5)-obj(iter-5+1:iter)))<thresh
        break;
    end
end
toc;
sumTime=toc;
end