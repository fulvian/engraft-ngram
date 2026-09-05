# engine check report

## Q1 (own overlay, per-fact)
- it_gatto: p_first=0.9594836684407282 rank_first=1 answer_reproduced=True hits=16
- it_mestiere: p_first=0.8477355618167294 rank_first=1 answer_reproduced=True hits=16
- it_citta: p_first=0.8996525362513614 rank_first=1 answer_reproduced=True hits=16
- it_capitale: p_first=0.020639931192355077 rank_first=9 answer_reproduced=False hits=16
- en_dog: p_first=0.9297818504683775 rank_first=1 answer_reproduced=True hits=16
- en_job: p_first=0.9347079255403454 rank_first=1 answer_reproduced=True hits=16
- en_city: p_first=0.9469566814576511 rank_first=1 answer_reproduced=True hits=16
- en_planet: p_first=0.9103610331622233 rank_first=1 answer_reproduced=True hits=16

## Q2/Q4 (merged overlay, corpus of all facts)
- it_gatto: p_first=0.9594836684407282 rank_first=1 answer_reproduced=True hits=16
    - paraphrase_same_tail: p_first=0.2813919612629984 rank_first=1 argmax_unchanged_vs_base=False delta_logp_argmax_base=-1.1932524355251113
    - paraphrase_other_tail: p_first=0.0005278183356774738 rank_first=294 argmax_unchanged_vs_base=True delta_logp_argmax_base=0.0
- en_dog: p_first=0.9297818504683775 rank_first=1 answer_reproduced=True hits=16
    - paraphrase_same_tail: p_first=0.0022862089277198763 rank_first=71 argmax_unchanged_vs_base=False delta_logp_argmax_base=-1.5965836076230566
    - paraphrase_other_tail: p_first=8.433330202831975e-05 rank_first=517 argmax_unchanged_vs_base=True delta_logp_argmax_base=0.0
- it_mestiere: p_first=0.8477355618167294 rank_first=1 answer_reproduced=True hits=16
    - paraphrase_same_tail: p_first=1.1974346111727707e-05 rank_first=2585 argmax_unchanged_vs_base=True delta_logp_argmax_base=0.39859157856637717
    - paraphrase_other_tail: p_first=8.079943923830077e-05 rank_first=872 argmax_unchanged_vs_base=True delta_logp_argmax_base=0.0
- en_job: p_first=0.9347079255403454 rank_first=1 answer_reproduced=True hits=16
    - paraphrase_same_tail: p_first=0.00039094696665879807 rank_first=423 argmax_unchanged_vs_base=True delta_logp_argmax_base=-0.05367246782914803
    - paraphrase_other_tail: p_first=3.871752531692918e-05 rank_first=1546 argmax_unchanged_vs_base=True delta_logp_argmax_base=0.0
- it_citta: p_first=0.8996525362513614 rank_first=1 answer_reproduced=True hits=16
    - paraphrase_same_tail: p_first=0.0018915553431767666 rank_first=79 argmax_unchanged_vs_base=True delta_logp_argmax_base=-0.10245500899879367
    - paraphrase_other_tail: p_first=0.0008466636179399308 rank_first=123 argmax_unchanged_vs_base=True delta_logp_argmax_base=0.0
- en_city: p_first=0.9469566814576511 rank_first=1 answer_reproduced=True hits=16
    - paraphrase_same_tail: p_first=2.5695963014026315e-05 rank_first=1485 argmax_unchanged_vs_base=True delta_logp_argmax_base=0.07044566216127168
    - paraphrase_other_tail: p_first=8.638170160341522e-05 rank_first=1232 argmax_unchanged_vs_base=True delta_logp_argmax_base=0.0
- it_capitale: p_first=0.020639931192355077 rank_first=9 answer_reproduced=False hits=16
    - paraphrase_same_tail: p_first=0.0025250468316443155 rank_first=16 argmax_unchanged_vs_base=True delta_logp_argmax_base=0.08638018741414982
    - paraphrase_other_tail: p_first=0.00042926365099163954 rank_first=87 argmax_unchanged_vs_base=True delta_logp_argmax_base=0.0
- en_planet: p_first=0.9103610331622233 rank_first=1 answer_reproduced=True hits=16
    - paraphrase_same_tail: p_first=0.0014610984202576456 rank_first=42 argmax_unchanged_vs_base=True delta_logp_argmax_base=-0.3425746357331055
    - paraphrase_other_tail: p_first=2.1054335072432457e-06 rank_first=937 argmax_unchanged_vs_base=True delta_logp_argmax_base=0.0

corpus: {'it': {'nll_base': 0.5017820236593887, 'nll_merged': 0.5017820236593887, 'delta_nll': 0.0, 'overlay_hits': 0}, 'en': {'nll_base': 0.04849951865102093, 'nll_merged': 0.04849951865102093, 'delta_nll': 0.0, 'overlay_hits': 0}}

docs (Q3, aggregate statistics):
- it: response={'n': 8, 'mean': 1.4948004486991389, 'min': -2.388422675143083, 'max': 12.98566375592338} other={'n': 39, 'mean': -0.1349497625887121, 'min': -2.9625723331683065, 'max': 1.0331652975786794} overlay_hits=128
- en: response={'n': 6, 'mean': -0.17304432216828344, 'min': -1.096386592393177, 'max': 0.9207132402701914} other={'n': 29, 'mean': -0.06494242152051004, 'min': -2.781006744074646, 'max': 0.8495825839540951} overlay_hits=96

### docs -- response positions
| lang | fid | position | target | logp_base | logp_merged | delta |
|---|---|---|---|---|---|---|
| it | it_gatto | 0 |  Ott | -6.4381 | -1.3417 | 5.0965 |
| it | it_gatto | 1 | av | -0.9297 | -2.4976 | -1.5679 |
| it | it_gatto | 2 | io | -0.0410 | -0.1165 | -0.0755 |
| it | it_mestiere | 0 |  li | -13.8086 | -0.8229 | 12.9857 |
| it | it_mestiere | 1 | uta | -0.2405 | -0.0052 | 0.2353 |
| it | it_mestiere | 2 | ia | -0.0007 | -2.3333 | -2.3326 |
| it | it_citta | 0 |  Rover | -5.9917 | -8.3801 | -2.3884 |
| it | it_citta | 1 | eto | -0.0101 | -0.0047 | 0.0054 |
| en | en_dog | 0 |  Pumpkin | -6.5046 | -5.5839 | 0.9207 |
| en | en_job | 0 |  glass | -9.4678 | -9.7135 | -0.2457 |
| en | en_job | 1 | bl | -0.1817 | -0.7804 | -0.5987 |
| en | en_job | 2 | ower | -0.0023 | -0.0013 | 0.0010 |
| en | en_city | 0 |  Dund | -10.2165 | -11.3129 | -1.0964 |
| en | en_city | 1 | ee | -0.0318 | -0.0511 | -0.0193 |

## Q5 (F32 fidelity)
- it_gatto_0: logp_base_f32=-7.083059899555824 p_y_free=0.9508572199017682 q5_pass=True diverging=0 consistency_pass=True
- it_gatto_1: logp_base_f32=-0.4159845055287344 p_y_free=0.9607188227424954 q5_pass=True diverging=0 consistency_pass=True
- it_gatto_2: logp_base_f32=-0.28165363254965026 p_y_free=0.9569900436247877 q5_pass=True diverging=0 consistency_pass=True
- it_mestiere_0: logp_base_f32=-12.684255621143537 p_y_free=0.9514809067057846 q5_pass=True diverging=0 consistency_pass=True
- it_mestiere_1: logp_base_f32=-0.01606369269255835 p_y_free=0.9840650153052267 q5_pass=True diverging=0 consistency_pass=True
- it_mestiere_2: logp_base_f32=-0.44413760520886636 p_y_free=0.9550802869032234 q5_pass=True diverging=0 consistency_pass=True
- it_citta_0: logp_base_f32=-6.179619895806236 p_y_free=0.9500448252479292 q5_pass=True diverging=0 consistency_pass=True
- it_citta_1: logp_base_f32=-0.0017200735601055862 p_y_free=0.9982813834929846 q5_pass=True diverging=0 consistency_pass=True
- it_capitale_0: logp_base_f32=-10.257869967813336 p_y_free=0.020252278141198725 q5_pass=True diverging=0 consistency_pass=True
- it_capitale_1: logp_base_f32=-0.005500696162335527 p_y_free=0.9945144474593903 q5_pass=True diverging=0 consistency_pass=True
- en_dog_0: logp_base_f32=-7.755967434286097 p_y_free=0.9803932159498321 q5_pass=True diverging=0 consistency_pass=True
- en_job_0: logp_base_f32=-10.751128126638552 p_y_free=0.9800127618476941 q5_pass=True diverging=0 consistency_pass=True
- en_job_1: logp_base_f32=-0.38426407810373825 p_y_free=0.966637018238906 q5_pass=True diverging=0 consistency_pass=True
- en_job_2: logp_base_f32=-0.0012856746741884168 p_y_free=0.9987151492937038 q5_pass=True diverging=0 consistency_pass=True
- en_city_0: logp_base_f32=-12.086281437801206 p_y_free=0.9548458344019923 q5_pass=True diverging=0 consistency_pass=True
- en_city_1: logp_base_f32=-0.07708116922738889 p_y_free=0.9530200594834959 q5_pass=True diverging=0 consistency_pass=True
- en_planet_0: logp_base_f32=-4.461050640267349 p_y_free=0.9582234537729434 q5_pass=True diverging=0 consistency_pass=True
