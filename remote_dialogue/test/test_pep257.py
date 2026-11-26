# Copyright 2015 Open Source Robotics Foundation, Inc.
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

from ament_pep257.main import main
import pytest


@pytest.mark.linter
@pytest.mark.pep257
def test_pep257():
    # Codes disabled deliberately, so this check stays meaningful rather than
    # permanently red:
    #   D213  conflicts with D212, which the "ament" convention already
    #         ignores; the pair is mutually exclusive and this repo puts the
    #         summary on the first line.
    #   D205/D400/D415
    #         assume a one-line summary. This repo wraps summaries across
    #         lines, so these fire on well-formed docstrings.
    #   D406/D407/D413
    #         are numpy-style section rules; this repo uses Google-style
    #         "Args:"/"Returns:" sections.
    #   D401  imperative mood — too many false positives on descriptive
    #         module and test docstrings.
    rc = main(argv=[
        '.', 'test',
        '--add-ignore',
        'D205', 'D213', 'D400', 'D401', 'D406', 'D407', 'D413', 'D415',
    ])
    assert rc == 0, 'Found code style errors / warnings'
