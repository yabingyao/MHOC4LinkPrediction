using LightGraphs
function CAR(graph,node1,node2)
    #计算CAR指标
    #CAR(x,y) = CN(x,y) * segema(s∈ T(x)andT(y))|gama(s)|/2
    #gama(s) is neighbor of s and common neighbors of x and y
    #intersect()返回两个集合的交集
    cn = common_neighbors(graph,node1,node2)
    tempvalue = 0
    for node in cn
        node_neighbors = neighbors(graph,node)
        tempvalue += length(intersect(Set(cn),Set(node_neighbors)))/2
    end
    return length(cn) * tempvalue
end

function CNAARA(graph,node1,node2)
    #graph简单图，node1和node2是候选连边顶点
    #返回三个值，分别是node1和node2的CN，AA，RA的相似性分值
    CN = 0
    AA = 0
    RA = 0
    CN = common_neighbors(graph,node1,nodw2)
    tempvec =  [1/log2(degree(graph,node)) for node in CN]
    #计算AA
    for x in tempvec
        if isinf(x)||isnan(x)
            continue
        else
            AA += x
        end
    end
    tempvalue =  [1/degree(graph,node) for node in CN]
    #计算RA
    for x in tempvalue
        if isinf(x)||isnan(x)
            continue
        else
            RA += x
        end
    end
    return length(CN),AA,RA
end
