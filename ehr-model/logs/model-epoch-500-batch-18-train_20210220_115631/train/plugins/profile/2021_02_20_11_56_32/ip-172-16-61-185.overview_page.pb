�	�:␍�?�:␍�?!�:␍�?	bA�d�)@bA�d�)@!bA�d�)@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$�:␍�?h�4�;�?A�r�SrN�?Y��8�	��?*	+���W@2F
Iterator::Model(��G��?!����*�G@)Y��+���?17�pÍ A@:Preprocessing2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat�*øD�?!/����;@)X��"�t�?1/��� 6@:Preprocessing2S
Iterator::Model::ParallelMap='�o|�?!�9�s�*@)='�o|�?1�9�s�*@:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMap�|[�T�?!�c�1�0@)����?1B!�~&@:Preprocessing2X
!Iterator::Model::ParallelMap::Zip:�<c_��?!UUUU�YJ@)�I����?1l���Z{ @:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor���W�<w?!     �@)���W�<w?1     �@:Preprocessing2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[0]::TensorSlice�A�t?!�RJ)�@)�A�t?1�RJ)�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 12.8% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*moderate2B10.9 % of the total step time sampled is spent on All Others time.>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	h�4�;�?h�4�;�?!h�4�;�?      ��!       "      ��!       *      ��!       2	�r�SrN�?�r�SrN�?!�r�SrN�?:      ��!       B      ��!       J	��8�	��?��8�	��?!��8�	��?R      ��!       Z	��8�	��?��8�	��?!��8�	��?JCPU_ONLY2black"�
both�Your program is MODERATELY input-bound because 12.8% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendationR
nomoderate"B10.9 % of the total step time sampled is spent on All Others time.:
Refer to the TF2 Profiler FAQ2"CPU: 