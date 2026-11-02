## Completed results

Frozen dataset: 2048 training / 512 validation patches; 26 validation subjects.
All adapters compared at update 65600. Original scorer has zero further updates.

|Scorer|Train conditional AP|Validation conditional AP|Validation all-retained AP|
|---|---:|---:|---:|
|original|0.4856|0.4723|0.1253|
|token_geometry|0.6044|0.3058|0.0777|
|endpoints|0.6394|0.3113|0.0833|
|chord|0.6468|0.3216|0.0882|
|corridor|0.6445|0.3211|0.0889|

|Image arm|Reference|Conditional AP difference: paired 95% interval|All-retained AP difference|
|---|---|---|---|
|endpoints|original|[-0.1762331077399786, -0.14446747881844924]|[-0.04868210569556284, -0.03484885482858969]|
|endpoints|token_geometry|[-0.014974921422470207, 0.027981053532356897]|[8.633121039858805e-05, 0.012093428127006096]|
|chord|original|[-0.1674168901888431, -0.1343847192597514]|[-0.044588013807320095, -0.02950155787975389]|
|chord|token_geometry|[-0.0029939481154729205, 0.03440075538025805]|[0.0055082280752560536, 0.01575520564066627]|
|corridor|original|[-0.16814313080522464, -0.13193146539731687]|[-0.04335689428747157, -0.029281941055504933]|
|corridor|token_geometry|[-0.009083713273264706, 0.044931597693531056]|[0.005563886317298051, 0.017100792907536477]|

|Scorer|Policy|Precision|Full GT recall|Edge F1|Branch F1|Curve F1|Component MAE|Cycle MAE|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|original|fixed_0p5|0.1554|0.0521|0.0780|0.0118|0.1732|16.2148|0.9941|
|original|train_max_f1|0.1202|0.2114|0.1533|0.0291|0.3874|2.7168|13.2949|
|original|train_preserve_precision|0.1571|0.0541|0.0805|0.0118|0.1770|16.0781|1.0156|
|token_geometry|fixed_0p5|0.0892|0.1234|0.1036|0.0189|0.3061|7.3633|11.2734|
|token_geometry|train_max_f1|0.0873|0.2117|0.1237|0.0264|0.3705|3.0605|26.0703|
|token_geometry|train_preserve_precision|0.1091|0.0095|0.0175|0.0047|0.0626|20.3594|0.6836|
|endpoints|fixed_0p5|0.0917|0.1255|0.1060|0.0216|0.3284|6.1680|9.7832|
|endpoints|train_max_f1|0.0916|0.1897|0.1236|0.0276|0.3710|3.1172|19.7402|
|endpoints|train_preserve_precision|0.0915|0.1193|0.1036|0.0204|0.3229|6.6133|8.9922|
|chord|fixed_0p5|0.1031|0.1487|0.1218|0.0258|0.3632|5.4570|10.4688|
|chord|train_max_f1|0.0984|0.2076|0.1335|0.0311|0.4044|2.8926|20.2598|
|chord|train_preserve_precision|0.1030|0.1367|0.1175|0.0252|0.3552|6.2109|9.0469|
|corridor|fixed_0p5|0.1024|0.1363|0.1169|0.0283|0.3621|5.4375|8.4922|
|corridor|train_max_f1|0.0998|0.1710|0.1261|0.0304|0.3911|3.5938|13.7051|
|corridor|train_preserve_precision|0.1001|0.1782|0.1282|0.0306|0.3944|3.4316|14.7480|

### Interpretation and limitations

Image arms with positive paired conditional-AP intervals against BOTH the frozen original and token/geometry control: none. This is exploratory evidence from one seed and reused validation subjects, with multiple comparisons and no multiplicity adjustment.
Compare training/validation AP and predefined intermediate checkpoints to distinguish fitting from generalization. If token-only gains match image gains, added capacity/training remains sufficient to explain the result. Failure of these adapters rules out this particular sampling/training design, not all image-aware relation learning.
Ranking AP is independent of the calibrated cutoff. Training-max-F1 and the common original training-precision policy use training labels only; their validation precision may differ. Validation-only equal-precision readouts are explicitly diagnostic. Fewer components and increased Curve F1 do not establish anatomical correctness when false cycles increase.
Recommendation: if an image arm improves validation ranking against both controls and its calibrated graph errors are acceptable, confirm the lightweight adapter before a larger transformer change. If it only improves fitting, examine site/generalization and label reliability. If no adapter improves ranking, retain the original scorer and review the relation representation/supervision before another architecture change. No multiscale or mask-input experiment was launched.

Supporting files: evaluation/*/{ranking.csv,ranking-strata.csv,per-subject.csv,per-site.csv,positive-lengths.csv,intermediate-ranking.json,validation-only-precision.json}; bootstrap JSONs; fixed 3D PNGs and sampling-point JSONs.
Full outputs: `/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/ixi_image_relation_learning_20261020-113444`.

