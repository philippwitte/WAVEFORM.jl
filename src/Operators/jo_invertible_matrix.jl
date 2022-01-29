function joInvertibleMatrix(A::AbstractMatrix{F};DDT::DataType=F,RDT::DataType=promote_type(F,DDT)) where {F<:Number}
    (m,n) = size(A)
    m==n || throw(ArgumentError("A must be square"))
    Fact = lu(A);
    (L,U,p,q,R) = Fact.L, Fact.U, Fact.p, Fact.q, Fact.Rs
    P = joPermutation(p,DDT=DDT)
    Q = joPermutation(q,DDT=DDT)
    Ut = U'
    Lt = L'
    forw_div = v->Q'*(U\(L\(P*(R.*v))))
    adj_div = v->conj(R).*(P'*(Lt\(Ut\(Q*v))))
    
    return joLinearFunction_A(n,n,
                              v->A*v,                               
                              v->A'*v,
                              forw_div,
                              adj_div,
                              DDT,RDT,name="joInvertibleMatrix",fMVok=true,iMVok=true)
end

