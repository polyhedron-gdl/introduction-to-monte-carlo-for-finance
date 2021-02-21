import datetime
#__________________________________________________________________________________
# 
# TEST 1
#
vol_times_test_1    = [ 0.498630137,
                        1.0054794521,
                        1.5068493151,
                        2.0082191781,
                        2.5068493151,
                        3.0082191781,
                        3.5068493151,
                        4.0054794521,
                        4.504109589,
                        5.0082191781,
                        5.5,
                        6.0,
                        6.5]
vol_value_test_1     = [0.0050387827,
                        0.0050545975,
                        0.0050632421,
                        0.0050458962,
                        0.0050291696,
                        0.0049970644,
                        0.0049700201,
                        0.0049313693,
                        0.0048949267,
                        0.0048949267,
                        0.0048949267,
                        0.0048949267,
                        0.0048949267,
                        ]

vol_value_test_2    =   [0.0052,
                         0.0052,
                         0.0052,
                         0.0052,
                         0.0052,
                         0.0052,
                         0.0052,
                         0.0052,
                         0.0052,
                         0.0055,
                         0.006,
                         0.006,
                         0.006,
                         ]
#__________________________________________________________________________________
# 
# TEST 2
#
check_dates_test_1   = [datetime.date(2014,11,1), datetime.date(2014,11,8), datetime.date(2014,11,15),datetime.date(2014,11,22),datetime.date(2014,11,29),datetime.date(2014,12,6),datetime.date(2014,12,13),datetime.date(2014,12,20),datetime.date(2014,12,27),datetime.date(2015,1,3),datetime.date(2015,1,10),datetime.date(2015,1,17),datetime.date(2015,1,24),datetime.date(2015,1,31),datetime.date(2015,2,7),datetime.date(2015,2,14),datetime.date(2015,2,21),datetime.date(2015,2,28),datetime.date(2015,3,7),datetime.date(2015,3,14),datetime.date(2015,3,21),datetime.date(2015,3,28),datetime.date(2015,4,4),datetime.date(2015,4,11),datetime.date(2015,4,18),datetime.date(2015,4,25),datetime.date(2015,5,2),datetime.date(2015,5,9),datetime.date(2015,5,16),datetime.date(2015,5,23),datetime.date(2015,5,30), datetime.date(2015,6,6),datetime.date(2015,6,13),datetime.date(2015,6,20),datetime.date(2015,6,27),datetime.date(2015,7,4),datetime.date(2015,7,11),datetime.date(2015,7,18),datetime.date(2015,7,25),datetime.date(2015,8,1), datetime.date(2015,8,8),datetime.date(2015,8,15), datetime.date(2015,8,22), datetime.date(2015,8,29), datetime.date(2015,9,5), datetime.date(2015,9,12),datetime.date(2015,9,19),datetime.date(2015,9,26),datetime.date(2015,10,3),datetime.date(2015,10,10),datetime.date(2015,10,17),datetime.date(2015,10,24),  datetime.date(2015,10,31), datetime.date(2015,11,7),datetime.date(2015,11,14),datetime.date(2015,11,21),datetime.date(2015,11,28),datetime.date(2015,12,5),datetime.date(2015,12,12), datetime.date(2015,12,19),datetime.date(2015,12,26),datetime.date(2016,1,2),datetime.date(2016,1,9),datetime.date(2016,1,16),datetime.date(2016,1,23),datetime.date(2016,1,30),datetime.date(2016,2,6),datetime.date(2016,2,13), datetime.date(2016,2,20),datetime.date(2016,2,27),datetime.date(2016,3,5), datetime.date(2016,3,12), datetime.date(2016,3,19), datetime.date(2016,3,26),datetime.date(2016,4,2), datetime.date(2016,4,9), datetime.date(2016,4,16),datetime.date(2016,4,23),datetime.date(2016,4,30), datetime.date(2016,5,7), datetime.date(2016,5,14),datetime.date(2016,5,21), datetime.date(2016,5,28), datetime.date(2016,6,4),datetime.date(2016,6,11), datetime.date(2016,6,18), datetime.date(2016,6,25),datetime.date(2016,7,2), datetime.date(2016,7,9), datetime.date(2016,7,16),datetime.date(2016,7,23), datetime.date(2016,7,30), datetime.date(2016,8,6), datetime.date(2016,8,13), datetime.date(2016,8,20), datetime.date(2016,8,27), datetime.date(2016,9,3),  datetime.date(2016,9,10),datetime.date(2016,9,17), datetime.date(2016,9,24),datetime.date(2016,10,1), datetime.date(2016,10,8),datetime.date(2016,10,15), datetime.date(2016,10,22),datetime.date(2016,10,29), datetime.date(2016,11,5),datetime.date(2016,11,12), datetime.date(2016,11,19), datetime.date(2016,11,26), datetime.date(2016,12,3),datetime.date(2016,12,10), datetime.date(2016,12,17),datetime.date(2016,12,24), datetime.date(2016,12,31),datetime.date(2017,1,7),  datetime.date(2017,1,14),datetime.date(2017,1,21), datetime.date(2017,1,28),  datetime.date(2017,2,4),datetime.date(2017,2,11),datetime.date(2017,2,18),datetime.date(2017,2,25),datetime.date(2017,3,4), datetime.date(2017,3,11), datetime.date(2017,3,18), datetime.date(2017,3,25),datetime.date(2017,4,1), datetime.date(2017,4,8),datetime.date(2017,4,15),datetime.date(2017,4,22), datetime.date(2017,4,29), datetime.date(2017,5,6), datetime.date(2017,5,13), datetime.date(2017,5,20),datetime.date(2017,5,27),datetime.date(2017,6,3),datetime.date(2017,6,10), datetime.date(2017,6,17), datetime.date(2017,6,24), datetime.date(2017,7,1), datetime.date(2017,7,8),datetime.date(2017,7,15), datetime.date(2017,7,22), datetime.date(2017,7,29),datetime.date(2017,8,5), datetime.date(2017,8,12), datetime.date(2017,8,19), datetime.date(2017,8,26),datetime.date(2017,9,2), datetime.date(2017,9,9), datetime.date(2017,9,16), datetime.date(2017,9,23),datetime.date(2017,9,30), datetime.date(2017,10,7),datetime.date(2017,10,14), datetime.date(2017,10,21), datetime.date(2017,10,28),  datetime.date(2017,11,4), datetime.date(2017,11,11), datetime.date(2017,11,18),datetime.date(2017,11,25), datetime.date(2017,12,2), datetime.date(2017,12,9), datetime.date(2017,12,16), datetime.date(2017,12,23), datetime.date(2017,12,30),datetime.date(2018,1,6), datetime.date(2018,1,13),datetime.date(2018,1,20), datetime.date(2018,1,27),datetime.date(2018,2,3), datetime.date(2018,2,10), datetime.date(2018,2,17), datetime.date(2018,2,24),datetime.date(2018,3,3), datetime.date(2018,3,10), datetime.date(2018,3,17),datetime.date(2018,3,24), datetime.date(2018,3,31),datetime.date(2018,4,7), datetime.date(2018,4,14),datetime.date(2018,4,21), datetime.date(2018,4,28), datetime.date(2018,5,5), datetime.date(2018,5,12), datetime.date(2018,5,19),datetime.date(2018,5,26), datetime.date(2018,6,2), datetime.date(2018,6,9),  datetime.date(2018,6,16),datetime.date(2018,6,23), datetime.date(2018,6,30),datetime.date(2018,7,7),  datetime.date(2018,7,14),datetime.date(2018,7,21), datetime.date(2018,7,28), datetime.date(2018,8,4),  datetime.date(2018,8,11),datetime.date(2018,8,18), datetime.date(2018,8,25),datetime.date(2018,9,1), datetime.date(2018,9,8), datetime.date(2018,9,15),datetime.date(2018,9,22), datetime.date(2018,9,29), datetime.date(2018,10,6),datetime.date(2018,10,13), datetime.date(2018,10,20),datetime.date(2018,10,27),datetime.date(2018,11,3),datetime.date(2018,11,10),datetime.date(2018,11,17),datetime.date(2018,11,24),datetime.date(2018,12,1),datetime.date(2018,12,8),datetime.date(2018,12,15),datetime.date(2018,12,22),datetime.date(2018,12,29),datetime.date(2019,1,5),datetime.date(2019,1,12),datetime.date(2019,1,19),datetime.date(2019,1,26),datetime.date(2019,2,2),datetime.date(2019,2,9),datetime.date(2019,2,16),datetime.date(2019,2,23),datetime.date(2019,3,2),datetime.date(2019,3,9),datetime.date(2019,3,16),datetime.date(2019,3,23),datetime.date(2019,3,30),datetime.date(2019,4,6),datetime.date(2019,4,13),datetime.date(2019,4,20),datetime.date(2019,4,27), datetime.date(2019,5,4),datetime.date(2019,5,11),datetime.date(2019,5,18),datetime.date(2019,5,25),datetime.date(2019,6,1),datetime.date(2019,6,8),datetime.date(2019,6,15),datetime.date(2019,6,22),datetime.date(2019,6,29),datetime.date(2019,7,6),datetime.date(2019,7,13),datetime.date(2019,7,20),datetime.date(2019,7,27),datetime.date(2019,8,3),datetime.date(2019,8,10), datetime.date(2019,8,17), datetime.date(2019,8,24), datetime.date(2019,8,31), datetime.date(2019,9,7), datetime.date(2019,9,14), datetime.date(2019,9,21),datetime.date(2019,9,28),datetime.date(2019,10,5),datetime.date(2019,10,12), datetime.date(2019,10,19), datetime.date(2019,10,26)]

check_dates_test_2   = [datetime.date(2014,10,24),
                        datetime.date(2014,11,24),
                        datetime.date(2014,12,24),
                        datetime.date(2015,1,24),
                        datetime.date(2015,2,24),
                        datetime.date(2015,3,24),
                        datetime.date(2015,4,24),
                        datetime.date(2015,5,24),
                        datetime.date(2015,6,24),
                        datetime.date(2015,7,24),
                        datetime.date(2015,8,24),
                        datetime.date(2015,9,24),
                        datetime.date(2015,10,24),
                        datetime.date(2015,10,26),
                        datetime.date(2015,11,24),
                        datetime.date(2015,12,24),
                        datetime.date(2016,1,24),
                        datetime.date(2016,2,24),
                        datetime.date(2016,3,24),
                        datetime.date(2016,4,24),
                        datetime.date(2016,4,26),
                        datetime.date(2016,5,24),
                        datetime.date(2016,6,24),
                        datetime.date(2016,7,24),
                        datetime.date(2016,8,24),
                        datetime.date(2016,9,24),
                        datetime.date(2016,10,24),
                        datetime.date(2016,10,26),
                        datetime.date(2016,11,24),
                        datetime.date(2016,12,24),
                        datetime.date(2017,1,24),
                        datetime.date(2017,2,24),
                        datetime.date(2017,3,24),
                        datetime.date(2017,4,24),
                        datetime.date(2017,4,26),
                        datetime.date(2017,5,24),
                        datetime.date(2017,6,24),
                        datetime.date(2017,7,24),
                        datetime.date(2017,8,24),
                        datetime.date(2017,9,24),
                        datetime.date(2017,10,24),
                        datetime.date(2017,10,26),
                        datetime.date(2017,11,24),
                        datetime.date(2017,12,24),
                        datetime.date(2018,1,24),
                        datetime.date(2018,2,24),
                        datetime.date(2018,3,24),
                        datetime.date(2018,4,24),
                        datetime.date(2018,4,26),
                        datetime.date(2018,5,24),
                        datetime.date(2018,6,24),
                        datetime.date(2018,7,24),
                        datetime.date(2018,8,24),
                        datetime.date(2018,9,24),
                        datetime.date(2018,10,24),
                        datetime.date(2018,10,25),
                        datetime.date(2018,11,24),
                        datetime.date(2018,12,24),
                        datetime.date(2019,1,24),
                        datetime.date(2019,2,24),
                        datetime.date(2019,3,24),
                        datetime.date(2019,4,24),
                        datetime.date(2019,4,25),
                        datetime.date(2019,5,24),
                        datetime.date(2019,6,24),
                        datetime.date(2019,7,24),
                        datetime.date(2019,8,24),
                        datetime.date(2019,9,24),
                        datetime.date(2019,10,24),
                        datetime.date(2019,11,24),
                        datetime.date(2019,12,24),
                        datetime.date(2020,1,24),
                        datetime.date(2020,2,24),
                        datetime.date(2020,3,24),
                        ]
# same set as previous but without fixing dates 
check_dates_test_3   = [datetime.date(2014,10,24),
                        datetime.date(2014,11,24),
                        datetime.date(2014,12,24),
                        datetime.date(2015,1,24),
                        datetime.date(2015,2,24),
                        datetime.date(2015,3,24),
                        datetime.date(2015,4,24),
                        datetime.date(2015,5,24),
                        datetime.date(2015,6,24),
                        datetime.date(2015,7,24),
                        datetime.date(2015,8,24),
                        datetime.date(2015,9,24),
                        datetime.date(2015,10,24),
                        datetime.date(2015,11,24),
                        datetime.date(2015,12,24),
                        datetime.date(2016,1,24),
                        datetime.date(2016,2,24),
                        datetime.date(2016,3,24),
                        datetime.date(2016,4,24),
                        datetime.date(2016,5,24),
                        datetime.date(2016,6,24),
                        datetime.date(2016,7,24),
                        datetime.date(2016,8,24),
                        datetime.date(2016,9,24),
                        datetime.date(2016,10,24),
                        datetime.date(2016,11,24),
                        datetime.date(2016,12,24),
                        datetime.date(2017,1,24),
                        datetime.date(2017,2,24),
                        datetime.date(2017,3,24),
                        datetime.date(2017,4,24),
                        datetime.date(2017,5,24),
                        datetime.date(2017,6,24),
                        datetime.date(2017,7,24),
                        datetime.date(2017,8,24),
                        datetime.date(2017,9,24),
                        datetime.date(2017,10,24),
                        datetime.date(2017,11,24),
                        datetime.date(2017,12,24),
                        datetime.date(2018,1,24),
                        datetime.date(2018,2,24),
                        datetime.date(2018,3,24),
                        datetime.date(2018,4,24),
                        datetime.date(2018,5,24),
                        datetime.date(2018,6,24),
                        datetime.date(2018,7,24),
                        datetime.date(2018,8,24),
                        datetime.date(2018,9,24),
                        datetime.date(2018,10,24),
                        datetime.date(2018,11,24),
                        datetime.date(2018,12,24),
                        datetime.date(2019,1,24),
                        datetime.date(2019,2,24),
                        datetime.date(2019,3,24),
                        datetime.date(2019,4,24),
                        datetime.date(2019,5,24),
                        datetime.date(2019,6,24),
                        datetime.date(2019,7,24),
                        datetime.date(2019,8,24),
                        datetime.date(2019,9,24),
                        datetime.date(2019,10,24),
                        datetime.date(2019,11,24),
                        datetime.date(2019,12,24),
                        datetime.date(2020,1,24),
                        datetime.date(2020,2,24),
                        datetime.date(2020,3,24),
                        ]
#__________________________________________________________________________________
# 
# TEST PRICING CAP HULL-WHITE
#
fixing_dates_test_cap = [
                    datetime.date(2011,9,22),
                    datetime.date(2012,3,22),
                    datetime.date(2012,9,22),
                    datetime.date(2013,3,22),
                    datetime.date(2013,9,22),
                    datetime.date(2014,3,22),
                    datetime.date(2014,9,22),
                    datetime.date(2015,3,22),
                    datetime.date(2015,9,22),
                    datetime.date(2016,3,22),
                    datetime.date(2016,9,22),
                    datetime.date(2017,3,22),
                    datetime.date(2017,9,22),
                    datetime.date(2018,3,22),
                    datetime.date(2018,9,22),
                    datetime.date(2019,3,22),
                    datetime.date(2019,9,22),
                    datetime.date(2020,3,22),
                    datetime.date(2020,9,22),
                    datetime.date(2021,3,22),
                    datetime.date(2021,9,22),
                    datetime.date(2022,3,22),
                    datetime.date(2022,9,22),
                    datetime.date(2023,3,22),
                    datetime.date(2023,9,22),
                    datetime.date(2024,3,22),
                    datetime.date(2024,9,22),
                    datetime.date(2025,3,22),
                    datetime.date(2025,9,22),
                    datetime.date(2026,3,22),
                    datetime.date(2026,9,22),
                    datetime.date(2027,3,22),
                    datetime.date(2027,9,22),
                    datetime.date(2028,3,22),
                    datetime.date(2028,9,22),
                    datetime.date(2029,3,22),
                    datetime.date(2029,9,22),
                    datetime.date(2030,3,22),
                    datetime.date(2030,9,22),
                    datetime.date(2031,3,22),
                    datetime.date(2031,9,22),
                    datetime.date(2032,3,22),
                    datetime.date(2032,9,22),
                    datetime.date(2033,3,22),
                    datetime.date(2033,9,22),
                    datetime.date(2034,3,22),
                    datetime.date(2034,9,22),
                    datetime.date(2035,3,22),
                    datetime.date(2035,9,22),
                    datetime.date(2036,3,22),
                    datetime.date(2036,9,22),
                    datetime.date(2037,3,22),
                    datetime.date(2037,9,22),
                    datetime.date(2038,3,22),
                    datetime.date(2038,9,22),
                    datetime.date(2039,3,22),
                    datetime.date(2039,9,22),
                    datetime.date(2040,3,22),
                    datetime.date(2040,9,22),
                    ]
vol_times_test_cap       = [ 
                    0.504109589041096,
                    1.0027397260274,
                    1.50684931506849,
                    2.0027397260274,
                    2.50684931506849,
                    3.0027397260274,
                    3.50684931506849,
                    4.0027397260274,
                    4.50684931506849,
                    5.00547945205479,
                    5.50958904109589,
                    6.00547945205479,
                    6.50958904109589,
                    7.00547945205479,
                    7.50958904109589,
                    8.00547945205479,
                    8.50958904109589,
                    9.00821917808219,
                    9.51232876712329,
                    10.0082191780822,
                    10.5123287671233,
                    11.0082191780822,
                    11.5123287671233,
                    12.0082191780822,
                    12.5123287671233,
                    13.0109589041096,
                    13.5150684931507,
                    14.0109589041096,
                    14.5150684931507,
                    15.0109589041096,
                    15.5150684931507,
                    16.0109589041096,
                    16.5150684931507,
                    17.013698630137,
                    17.5178082191781,
                    18.013698630137,
                    18.5178082191781,
                    19.013698630137,
                    19.5178082191781,
                    20.013698630137,
                    20.5178082191781,
                    21.0164383561644,
                    21.5205479452055,
                    22.0164383561644,
                    22.5205479452055,
                    23.0164383561644,
                    23.5205479452055,
                    24.0164383561644,
                    24.5205479452055,
                    25.0191780821918,
                    25.5232876712329,
                    26.0191780821918,
                    26.5232876712329,
                    27.0191780821918,
                    27.5232876712329,
                    28.0191780821918,
                    28.5232876712329,
                    29.0219178082192,
                    29.5260273972603,
                      ]
vol_value_test_cap       = [
                    0.266294860740285,
                    0.321666955556033,
                    0.328891510003864,
                    0.332827727294911,
                    0.344256982127267,
                    0.320285048562887,
                    0.299426445576254,
                    0.283245597315139,
                    0.275405465421936,
                    0.261943108332192,
                    0.252206998915237,
                    0.237899939226548,
                    0.230683994203384,
                    0.225309043539597,
                    0.222024798634462,
                    0.210159904422869,
                    0.202964801234895,
                    0.2001132285516,
                    0.196170136438742,
                    0.186629472065581,
                    0.18169343502523,
                    0.176035390875826,
                    0.173630872416456,
                    0.171965484181573,
                    0.172359394726292,
                    0.172541697270548,
                    0.172232751142879,
                    0.174203978234153,
                    0.173516270816765,
                    0.17646661813109,
                    0.177018877187154,
                    0.179558316205875,
                    0.180282486424787,
                    0.183209713570859,
                    0.184416952487409,
                    0.186340825165367,
                    0.188169988902016,
                    0.190674471951513,
                    0.191640476335426,
                    0.19548128393875,
                    0.196313056222051,
                    0.200859253685811,
                    0.202663832069113,
                    0.206502579468537,
                    0.208398529067103,
                    0.212790087331329,
                    0.214823692463886,
                    0.217538256650008,
                    0.219128809666503,
                    0.222787060046281,
                    0.224055607895101,
                    0.227385126562144,
                    0.228664806558327,
                    0.231033077104592,
                    0.232024900226833,
                    0.236457990242272,
                    0.23693965617295,
                    0.239487599714963,
                    0.239772218114555,
                    ]

fixing_dates_gpp_cap = [
                    datetime.date(2011,9,22),
                    datetime.date(2012,3,22),
                    datetime.date(2012,9,20),
                    datetime.date(2013,3,21),
                    datetime.date(2013,9,20),
                    datetime.date(2014,3,20),
                    datetime.date(2014,9,22),
                    datetime.date(2015,3,20),
                    datetime.date(2015,9,22),
                    datetime.date(2016,3,22),
                    datetime.date(2016,9,22),
                    datetime.date(2017,3,22),
                    datetime.date(2017,9,21),
                    datetime.date(2018,3,22),
                    datetime.date(2018,9,20),
                    datetime.date(2019,3,21),
                    datetime.date(2019,9,20),
                    datetime.date(2020,3,20),
                    datetime.date(2020,9,22),
                    datetime.date(2021,3,22),
                    datetime.date(2021,9,22),
                    datetime.date(2022,3,22),
                    datetime.date(2022,9,22),
                    datetime.date(2023,3,22),
                    datetime.date(2023,9,21),
                    datetime.date(2024,3,21),
                    datetime.date(2024,9,20),
                    datetime.date(2025,3,20),
                    datetime.date(2025,9,22),
                    datetime.date(2026,3,20),
                    datetime.date(2026,9,22),
                    datetime.date(2027,3,22),
                    datetime.date(2027,9,22),
                    datetime.date(2028,3,22),
                    datetime.date(2028,9,21),
                    datetime.date(2029,3,22),
                    datetime.date(2029,9,20),
                    datetime.date(2030,3,21),
                    datetime.date(2030,9,20),
                    datetime.date(2031,3,20),
                    datetime.date(2031,9,22),
                    datetime.date(2032,3,22),
                    datetime.date(2032,9,22),
                    datetime.date(2033,3,22),
                    datetime.date(2033,9,22),
                    datetime.date(2034,3,22),
                    datetime.date(2034,9,21),
                    datetime.date(2035,3,21),
                    datetime.date(2035,9,20),
                    datetime.date(2036,3,20),
                    datetime.date(2036,9,22),
                    datetime.date(2037,3,20),
                    datetime.date(2037,9,22),
                    datetime.date(2038,3,22),
                    datetime.date(2038,9,22),
                    datetime.date(2039,3,22),
                    datetime.date(2039,9,22),
                    datetime.date(2040,3,22),
                       ]
vol_times_gpp_cap    = [ 
                        0.504109589041096,
                        1.0027397260274,
                        1.5013698630137,
                        2,
                        2.5013698630137,
                        2.9972602739726,
                        3.50684931506849,
                        3.9972602739726,
                        4.50684931506849,
                        5.00547945205479,
                        5.50958904109589,
                        6.00547945205479,
                        6.50684931506849,
                        7.00547945205479,
                        7.5041095890411,
                        8.0027397260274,
                        8.5041095890411,
                        9.0027397260274,
                        9.51232876712329,
                        10.0082191780822,
                        10.5123287671233,
                        11.0082191780822,
                        11.5123287671233,
                        12.0082191780822,
                        12.5095890410959,
                        13.0082191780822,
                        13.5095890410959,
                        14.0054794520548,
                        14.5150684931507,
                        15.0054794520548,
                        15.5150684931507,
                        16.0109589041096,
                        16.5150684931507,
                        17.013698630137,
                        17.5150684931507,
                        18.013698630137,
                        18.5123287671233,
                        19.0109589041096,
                        19.5123287671233,
                        20.0082191780822,
                        20.5178082191781,
                        21.0164383561644,
                        21.5205479452055,
                        22.0164383561644,
                        22.5205479452055,
                        23.0164383561644,
                        23.5178082191781,
                        24.013698630137,
                        24.5150684931507,
                        25.013698630137,
                        25.5232876712329,
                        26.013698630137,
                        26.5232876712329,
                        27.0191780821918,
                        27.5232876712329,
                        28.0191780821918,
                        28.5232876712329,
                        29.0219178082192,
                       ]
vol_value_gpp_cap    = [
0.00648335751172735,
0.0120699760187064,
0.0150251284559503,
0.0173730235003004,
0.0181900685122233,
0.0195421967126777,
0.0160570854161711,
0.0206163236159758,
0.0189209271582339,
0.0209672707427839,
0.0192774426058899,
0.0198465164383443,
0.0197179889438035,
0.0220445640142862,
0.0213757153429723,
0.0198449432931629,
0.0198268127334065,
0.0235590182273567,
0.0214348176895455,
0.022183361053038,
0.0211637430561251,
0.0227299950712735,
0.0224486988033022,
0.0235032035324729,
0.0242473943769469,
0.0237577362841407,
0.0244487043591932,
0.0256311368570669,
0.0250000811674257,
0.0255777895607899,
0.0261338184455448,
0.0255219249509535,
0.0268667074847062,
0.0257807940303413,
0.0276669728406318,
0.025138736245108,
0.0280439319210403,
0.0252480234163216,
0.0277643520282466,
0.0265792193894081,
0.0279315249285812,
0.0258485447116778,
0.0285110852704466,
0.0265610147036813,
0.0288699378972145,
0.0248996903990879,
0.0288268310893881,
0.0266287768606826,
0.0286021677258301,
0.0248301999426313,
0.028043690770619,
0.0262547035858975,
0.0280663021683229,
0.0270775593205402,
0.0283323908441709,
0.0260527254760177,
0.0281483164915895,
0.0267532657786618,
                       ]

discount_dates_test_cap  = [
                    datetime.date(2011,3,23),
                    datetime.date(2011,3,24),
                    datetime.date(2011,3,31),
                    datetime.date(2011,4,7),
                    datetime.date(2011,4,26),
                    datetime.date(2011,5,24),
                    datetime.date(2011,6,24),
                    datetime.date(2011,9,15),
                    datetime.date(2011,12,21),
                    datetime.date(2012,3,21),
                    datetime.date(2012,6,21),
                    datetime.date(2012,9,20),
                    datetime.date(2012,12,19),
                    datetime.date(2013,3,19),
                    datetime.date(2013,6,20),
                    datetime.date(2014,3,24),
                    datetime.date(2015,3,24),
                    datetime.date(2016,3,24),
                    datetime.date(2017,3,24),
                    datetime.date(2018,3,26),
                    datetime.date(2019,3,25),
                    datetime.date(2020,3,24),
                    datetime.date(2021,3,24),
                    datetime.date(2023,3,24),
                    datetime.date(2026,3,24),
                    datetime.date(2031,3,24),
                    datetime.date(2036,3,24),
                    datetime.date(2041,3,25),
                    datetime.date(2051,3,24),
                    datetime.date(2061,3,24),
                    ]
discount_values_test_cap = [
                    0.999983611379705,
                    0.999966111972745,
                    0.999826136313661,
                    0.999666767312977,
                    0.999196730490268,
                    0.9983252534936,
                    0.997112155403501,
                    0.993535038142404,
                    0.988666010251979,
                    0.983525926180688,
                    0.97784707926784,
                    0.971756063709607,
                    0.965362305225734,
                    0.958669593128705,
                    0.951571182862157,
                    0.926372554513647,
                    0.895161333555141,
                    0.862670635537479,
                    0.829909817552386,
                    0.797225612234529,
                    0.765268601076758,
                    0.734052642206441,
                    0.70316933510932,
                    0.64256148614836,
                    0.560287247456319,
                    0.454183092160855,
                    0.381306727061861,
                    0.33003354494644,
                    0.249461185431812,
                    0.183856481052712,
                            ]
 

forecast_dates_test_cap  = [
                    datetime.date(2011,4,26),
                    datetime.date(2011,5,24),
                    datetime.date(2011,6,24),
                    datetime.date(2011,7,25),
                    datetime.date(2011,8,24),
                    datetime.date(2011,9,26),
                    datetime.date(2011,10,24),
                    datetime.date(2011,11,24),
                    datetime.date(2011,12,27),
                    datetime.date(2012,1,24),
                    datetime.date(2012,2,24),
                    datetime.date(2012,3,26),
                    datetime.date(2012,4,24),
                    datetime.date(2012,5,24),
                    datetime.date(2012,6,25),
                    datetime.date(2012,7,24),
                    datetime.date(2012,8,24),
                    datetime.date(2012,9,24),
                    datetime.date(2013,3,25),
                    datetime.date(2014,3,24),
                    datetime.date(2015,3,24),
                    datetime.date(2016,3,24),
                    datetime.date(2017,3,24),
                    datetime.date(2018,3,26),
                    datetime.date(2019,3,25),
                    datetime.date(2020,3,24),
                    datetime.date(2021,3,24),
                    datetime.date(2022,3,24),
                    datetime.date(2023,3,24),
                    datetime.date(2024,3,25),
                    datetime.date(2025,3,24),
                    datetime.date(2026,3,24),
                    datetime.date(2027,3,24),
                    datetime.date(2028,3,24),
                    datetime.date(2029,3,26),
                    datetime.date(2030,3,25),
                    datetime.date(2031,3,24),
                    datetime.date(2032,3,24),
                    datetime.date(2033,3,24),
                    datetime.date(2034,3,24),
                    datetime.date(2035,3,26),
                    datetime.date(2036,3,24),
                    datetime.date(2037,3,24),
                    datetime.date(2038,3,24),
                    datetime.date(2039,3,24),
                    datetime.date(2040,3,26),
                    datetime.date(2041,3,25),
                    datetime.date(2046,3,26),
                    datetime.date(2051,3,24),
                    datetime.date(2061,3,24),
                            ]
forecast_values_test_cap = [
                    0.998740450266699,
                    0.997693493071839,
                    0.996502291638803,
                    0.995104798892988,
                    0.993781722257694,
                    0.99234497880168,
                    0.990661550287354,
                    0.988979049236228,
                    0.987168612408677,
                    0.98546164607791,
                    0.983691019779692,
                    0.98196938122193,
                    0.979872662354042,
                    0.977909708973164,
                    0.975810851803053,
                    0.973764783594952,
                    0.971675390047601,
                    0.969645510321637,
                    0.955773865080979,
                    0.926459237807753,
                    0.895245096327225,
                    0.862751358063262,
                    0.829998294435987,
                    0.797310604647205,
                    0.765351951982524,
                    0.734133470626535,
                    0.703246762889218,
                    0.672609270501351,
                    0.642621498762735,
                    0.613570675699049,
                    0.586331095455665,
                    0.560358433132841,
                    0.536051026290416,
                    0.513297961688196,
                    0.492068529457721,
                    0.472551779651386,
                    0.454496107868845,
                    0.437575519811354,
                    0.422017702733181,
                    0.407529813458822,
                    0.394186843348968,
                    0.381761732818965,
                    0.370276843762532,
                    0.35949831515687,
                    0.349240431761905,
                    0.33958296147162,
                    0.330509489538051,
                    0.288277412169738,
                    0.249700350971474,
                    0.184033211684749,
                    ]
