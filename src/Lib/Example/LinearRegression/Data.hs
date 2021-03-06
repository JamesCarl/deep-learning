{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE OverloadedLists #-}
module Lib.Example.LinearRegression.Data where
import qualified Data.Vector                   as V

type TimeType = Float

data SizeMbTimeSec = SizeMbTimeSec { sizeMB:: V.Vector Float , timeSec:: V.Vector TimeType}
trainData :: SizeMbTimeSec
trainData = SizeMbTimeSec
    { sizeMB  = [ 0.7999435
                , 0.86922085
                , 0.10718739
                , 0.33798838
                , 0.60347974
                , 0.53500545
                , 0.30512893
                , 0.65360814
                , 0.1993751
                , 4.3608785e-2
                , 0.79991806
                , 0.72412884
                , 7.012278e-2
                , 0.8442899
                , 0.5001625
                , 0.30187398
                , 0.4996097
                , 0.44204944
                , 0.69130975
                , 0.25066292
                , 0.7089014
                , 0.80831605
                , 0.63817734
                , 0.62400746
                , 0.4945851
                , 0.74446267
                , 0.1447087
                , 0.24669564
                , 0.3453141
                , 0.32391566
                , 0.14679408
                , 0.61134696
                , 0.66146106
                , 0.28712738
                , 0.23805332
                , 0.44335395
                , 0.69372004
                , 0.26698816
                , 0.1490174
                , 0.57959056
                , 0.9101107
                , 0.17110509
                , 0.303616
                , 0.49784505
                , 0.53354734
                , 0.99967074
                , 0.91864127
                , 0.54469293
                , 0.12060791
                , 0.85927814
                , 0.39630353
                , 0.2703567
                , 0.5540334
                , 0.6745911
                , 0.515607
                , 0.34710997
                , 4.9713492e-2
                , 0.60366535
                , 0.30905604
                , 0.115192056
                , 0.10765374
                , 0.79885054
                , 0.58347166
                , 0.5926126
                , 0.30029905
                , 0.13075244
                , 7.981908e-2
                , 0.6247058
                , 0.1303792
                , 0.3208667
                , 0.28793055
                , 0.23450112
                , 0.80858886
                , 0.89246553
                , 0.24624759
                , 0.8173177
                , 0.7112838
                , 0.11555952
                , 0.36830205
                , 0.56654286
                , 2.653867e-2
                , 0.94049054
                , 0.7153843
                , 0.7958241
                , 7.7687144e-2
                , 0.8042487
                , 0.24793547
                , 0.44066226
                , 0.97141093
                , 0.38429457
                , 0.9145181
                , 0.43203348
                , 0.25641108
                , 9.6490264e-2
                , 0.1811626
                , 0.7345831
                , 0.5291938
                , 0.9808153
                , 0.93809783
                , 0.68960327
                ]
    , timeSec = [ 8.098424
                , 9.859865
                , 10.331388
                , 9.697914
                , 10.331204
                , 8.323465
                , 8.304881
                , 8.6849165
                , 9.545427
                , 10.581436
                , 8.932811
                , 9.667332
                , 8.275962
                , 8.649912
                , 10.417122
                , 8.72864
                , 8.881492
                , 9.309477
                , 8.613172
                , 10.410737
                , 10.991287
                , 8.836339
                , 10.278866
                , 9.091362
                , 10.165595
                , 10.610301
                , 9.483838
                , 10.252721
                , 10.0422325
                , 9.19863
                , 8.136764
                , 9.346241
                , 8.581008
                , 10.674321
                , 9.403644
                , 8.001262
                , 10.771856
                , 8.169222
                , 9.9520035
                , 8.892649
                , 8.729118
                , 8.582624
                , 10.095998
                , 8.68763
                , 10.261923
                , 8.346957
                , 8.705555
                , 8.178328
                , 10.842712
                , 9.608037
                , 10.489033
                , 9.299994
                , 10.409305
                , 10.774183
                , 10.958311
                , 10.904007
                , 9.308371
                , 9.564395
                , 9.957057
                , 10.68167
                , 10.575035
                , 9.57173
                , 9.616956
                , 10.298044
                , 10.797202
                , 10.367228
                , 8.560775
                , 9.210349
                , 10.37883
                , 9.960355
                , 8.994696
                , 10.788715
                , 10.918774
                , 10.566075
                , 9.925214
                , 10.002563
                , 10.2182665
                , 8.4455385
                , 8.1283655
                , 8.31396
                , 8.207571
                , 9.847546
                , 10.905305
                , 9.567315
                , 9.093852
                , 9.000578
                , 8.1132555
                , 10.488562
                , 10.15446
                , 9.545567
                , 9.262574
                , 8.5123205
                , 10.672793
                , 8.650698
                , 9.441508
                , 8.61672
                , 9.517661
                , 8.827084
                , 8.957394
                , 10.956856
                ]
    }

testData :: SizeMbTimeSec
testData = SizeMbTimeSec
    { sizeMB  = [ 3.2808006e-2
                , 0.619955
                , 0.7771296
                , 0.5659715
                , 0.7770682
                , 0.10782194
                , 0.10162717
                , 0.2283054
                , 0.5151426
                , 0.8604785
                , 0.31093705
                , 0.5557772
                , 9.198731e-2
                , 0.21663743
                , 0.8057072
                , 0.24287999
                , 0.2938307
                , 0.43649215
                , 0.2043904
                , 0.8035789
                , 0.9970956
                , 0.27877957
                , 0.7596219
                , 0.36378735
                , 0.72186494
                , 0.8701004
                , 0.49461257
                , 0.75090694
                , 0.68074405
                , 0.39954346
                , 4.5587838e-2
                , 0.4487471
                , 0.19366932
                , 0.89144045
                , 0.46788126
                , 4.2068958e-4
                , 0.9239521
                , 5.6407392e-2
                , 0.65066797
                , 0.29754943
                , 0.24303943
                , 0.19420815
                , 0.6986659
                , 0.22920996
                , 0.75397426
                , 0.1156525
                , 0.23518503
                , 5.944252e-2
                , 0.9475709
                , 0.5360124
                , 0.82967776
                , 0.4333316
                , 0.8031016
                , 0.92472786
                , 0.9861038
                , 0.9680022
                , 0.43612367
                , 0.5214649
                , 0.6523524
                , 0.8938901
                , 0.85834515
                , 0.5239099
                , 0.5389853
                , 0.7660149
                , 0.9324007
                , 0.7890759
                , 0.18692482
                , 0.40344977
                , 0.7929434
                , 0.6534515
                , 0.3315652
                , 0.929572
                , 0.9729244
                , 0.8553584
                , 0.64173806
                , 0.66752106
                , 0.7394221
                , 0.14851296
                , 4.2788386e-2
                , 0.10465342
                , 6.9190204e-2
                , 0.61584854
                , 0.96843505
                , 0.52243847
                , 0.36461723
                , 0.33352607
                , 3.7751973e-2
                , 0.8295206
                , 0.7181533
                , 0.5151887
                , 0.4208581
                , 0.17077339
                , 0.8909311
                , 0.2168991
                , 0.48050267
                , 0.20557344
                , 0.505887
                , 0.27569443
                , 0.3191313
                , 0.98561865
                ]
    , timeSec = [ 10.607662
                , 8.321562
                , 9.013966
                , 9.810439
                , 9.605017
                , 8.915387
                , 9.960825
                , 8.598125
                , 8.130826
                , 10.399754
                , 10.172386
                , 8.210368
                , 10.532869
                , 9.500487
                , 8.905622
                , 9.498829
                , 9.326148
                , 10.073929
                , 8.751988
                , 10.126704
                , 10.424948
                , 9.914532
                , 9.872023
                , 9.483755
                , 10.233388
                , 8.434126
                , 8.740087
                , 9.035942
                , 8.971747
                , 8.440382
                , 9.834041
                , 9.984383
                , 8.8613825
                , 8.71416
                , 9.330062
                , 10.08116
                , 8.800964
                , 8.447052
                , 9.738771
                , 10.730332
                , 8.513315
                , 8.910848
                , 9.493535
                , 9.600642
                , 10.999012
                , 10.755924
                , 9.634079
                , 8.361824
                , 10.577834
                , 9.1889105
                , 8.81107
                , 9.6621
                , 10.023773
                , 9.546821
                , 9.04133
                , 8.14914
                , 9.810996
                , 8.927168
                , 8.345576
                , 8.322961
                , 10.396551
                , 9.750415
                , 9.777838
                , 8.900897
                , 8.392258
                , 8.239457
                , 9.874117
                , 8.391138
                , 8.9626
                , 8.863791
                , 8.703504
                , 10.425766
                , 10.677397
                , 8.738743
                , 10.451953
                , 10.133852
                , 8.346679
                , 9.104906
                , 9.699629
                , 8.079616
                , 10.821472
                , 10.1461525
                , 10.387472
                , 8.233062
                , 10.412746
                , 8.743807
                , 9.321987
                , 10.914232
                , 9.152884
                , 10.743554
                , 9.296101
                , 8.769234
                , 8.289471
                , 8.543488
                , 10.203749
                , 9.587582
                , 10.942446
                , 10.814293
                , 10.0688095
                , 10.399831
                ]
    }
