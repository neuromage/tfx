# Copyright 2019 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TFX publisher."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from typing import Dict, List, Text

from tfx.orchestration import metadata
from tfx.utils import types


class Publisher(object):
  """Publish execution to metadata.

  Attributes:
    _metadata_handler: An instance of Metadata.
  """

  def __init__(self, metadata_handler: metadata.Metadata):
    self._metadata_handler = metadata_handler

  def publish_execution(self, execution_id: int,
                        input_dict: Dict[Text, List[types.TfxArtifact]],
                        output_dict: Dict[Text, List[types.TfxArtifact]],
                        use_cached_results: bool
                       ) -> Dict[Text, List[types.TfxArtifact]]:
    """Publishes a component execution to metadata.

    This function will do two things:
    1. update the execution that was previously registered before execution to
       complete or skipped state, depending on whether cached results are used.
    2. for each input and output artifact, publish an event that associate the
       artifact to the execution, with type INPUT or OUTPUT respectively

    Args:
      execution_id: the execution id for the
      input_dict: key -> Artifacts that are used as inputs in the execution
      output_dict: key -> Artifacts that are declared as outputs for the
        execution
      use_cached_results: whether or not the execution has used cached results

    Returns:
      A dict containing output artifacts.
    """
    tf.logging.info('Whether cached results are used: %s', use_cached_results)
    tf.logging.info('Execution id: %s', execution_id)
    tf.logging.info('Inputs: %s', input_dict)
    tf.logging.info('Outputs: %s', output_dict)

    final_execution_state = metadata.EXECUTION_STATE_CACHED if use_cached_results else metadata.EXECUTION_STATE_COMPLETE
    return self._metadata_handler.publish_execution(
        execution_id=execution_id,
        input_dict=input_dict,
        output_dict=output_dict,
        state=final_execution_state)
