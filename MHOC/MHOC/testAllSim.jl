using DataFrames
using CSV

include("read_undir_txt.jl")
include("similarity_index.jl")
include("DivideNet_notconnected.jl")
include("DivideNet.jl")
include("AUC.jl")
include("Calaupr.jl")
include("HigherOrderMI.jl")
include("CalMI.jl")
include("CalCNDP")
    #datasets = ["Karate_34_78.txt","USAir_332_2126.txt","NetScience_379_914.txt","celegans_297_2148.txt"]
    #datasets = ["Karate_34_78.txt","USAir_332_2126.txt","NetScience_379_914.txt","celegans_297_2148.txt","arenas-email_1133_5451.txt","blogs_1222_16714.txt"]
    #datasets = ["arenas-email_1133_5451.txt","blogs_1222_16714.txt","Yeast_2375_23386.txt"]
    #datasets = ["Karate_34_78.txt"]
    #datasets = ["arenas-email_1133_5451.txt"]
    #datasets = ["blogs_1222_16714.txt"]
    #datasets = ["Yeast_2375_23386.txt"]
    datasets = ["CAG_mat_364_10474.txt","c-fat_500_9139.txt","moreno_names_names_1707_9059.txt","socfb-Caltech_762_16651.txt"]
    counts = 10
    TopL = 100
    segama = 0.0001
    n = 670000
    cnpre = zeros(Float64,counts)
    homi3pre = zeros(Float64,counts)
    homi4pre = zeros(Float64,counts)
    homi23pre = zeros(Float64,counts)
    homi234pre = zeros(Float64,counts)
    homi3auc = zeros(Float64,counts)
    homi4auc = zeros(Float64,counts)
    homi23auc = zeros(Float64,counts)
    homi234auc = zeros(Float64,counts)
    raauc = zeros(Float64,counts)
    cnauc = zeros(Float64,counts)
    aaauc = zeros(Float64,counts)
    miauc = zeros(Float64,counts)
    paauc = zeros(Float64,counts)

    aapre = zeros(Float64,counts)
    rapre = zeros(Float64,counts)
    mipre = zeros(Float64,counts)
    papre = zeros(Float64,counts)
    df1 = DataFrame()
    df2 = DataFrame()

    for j = 1 : length(datasets)
        println("正在加载数据集",datasets[j])
        data = load_example_data(datasets[j])
        println("数据集加载完成")
        for i = 1 : counts
            println("第",i,"次")
            #println("开始划分训练集和测试集")
            net = DivideNetNoCon(data,0.9)
            #println("训练集划分结束")
            train = net[1]
            test = net[2]
            #println("开始计算相似性指标")
            println("HigherOrderMI相似性指标计算开始")
            homi3sim = CalHigherOrderMI(train,3)
            println("3OrderMI相似性指标计算结束")
            homi4sim = CalHigherOrderMI(train,4)
            println("4OrderMI相似性指标计算结束")
            homi23sim = CalHigherOrderMI(train,23)
            println("2-3OrderMI相似性指标计算结束")
            homi234sim = CalHigherOrderMI(train,234)
            println("2-3-4OrderMI相似性指标计算结束")
            println("MI相似性指标计算开始")
            misim = MI(train)
            println("MI相似性指标计算结束")
            #println("CNAARAPA相似性指标计算开始")
            cnsim = CN(train)
            #println("cn相似性指标计算结束")
            aasim = AA(train)
            #println("aa相似性指标计算结束")
            rasim = RA(train)
            #println("ra相似性指标计算结束")
            pasim = PA(train)
            #println("pa相似性指标计算结束")
            pasim = PA(train)
            #println("pa相似性指标计算结束")

            println("开始计算AUC指标")
            homi3auc[i]=CalAUC(train,test,homi3sim,n)
            homi4auc[i]=CalAUC(train,test,homi4sim,n)
            homi23auc[i]=CalAUC(train,test,homi23sim,n)
            homi234auc[i]=CalAUC(train,test,homi234sim,n)
            miauc[i]=CalAUC(train,test,misim,n)
            cnauc[i]=CalAUC(train,test,cnsim,n)
            raauc[i]=CalAUC(train,test,rasim,n)
            aaauc[i]=CalAUC(train,test,aasim,n)
            paauc[i]=CalAUC(train,test,pasim,n)
            #println("AUC指标计算结束")
            df1.CNauc = cnauc
            df1.AAauc = aaauc
            df1.RAauc = raauc
            df1.PAau = paauc
            df1.Order3MIauc = homi3auc
            df1.Order4MIauc = homi4auc
            df1.Order23MIauc = homi23auc
            df1.Order234MIauc = homi234auc
            df1.MIauc = miauc
            CSV.write(".\\result\\HigherOrder10MI$(datasets[j])auc.csv", df1)
            println("开始计算aupr指标")
            homi3pre[i] = CalPresion(train,test,homi3sim,TopL)
            homi4pre[i] = CalPresion(train,test,homi4sim,TopL)
            homi23pre[i] = CalPresion(train,test,homi23sim,TopL)
            homi234pre[i] = CalPresion(train,test,homi234sim,TopL)
            mipre[i] = CalPresion(train,test,misim,TopL)
            cnpre[i] = CalPresion(train,test,cnsim,TopL)
            aapre[i] = CalPresion(train,test,aasim,TopL)
            rapre[i] = CalPresion(train,test,rasim,TopL)
            papre[i] = CalPresion(train,test,pasim,TopL)
            df2.CNpre = cnpre
            df2.AApre = aapre
            df2.RApre = rapre
            df2.PApre = papre
            df2.Order3MIpre = homi3pre
            df2.Order4MIpre = homi4pre
            df2.Order23MIpre = homi23pre
            df2.Order234MIpre = homi234pre
            df2.MIpre = mipre
                #CSV.write(".\\data\\HigherOrderMIemailpre.csv", df2)
            CSV.write(".\\result\\HigherOrder10MI$(datasets[j])pre.csv", df2)
            println("aupr计算结束")
        end
            #df[1]表示数据框df中第1列的数据和变量=df[:var1]var1表示列名
    end
