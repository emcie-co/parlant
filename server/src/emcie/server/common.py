# Copyright (c) 2024 Emcie
# All rights reserved.
#
# This file and its contents are the property of Emcie and are strictly confidential.
# No part of this file may be reproduced, distributed, or transmitted in any form or by any means,
# including photocopying, recording, or other electronic or mechanical methods,
# without the prior written permission of Emcie.
#
# Website: https://emcie.co
from typing import NewType
import nanoid  # type: ignore

UniqueId = NewType("UniqueId", str)


def generate_id() -> UniqueId:
    return UniqueId(nanoid.generate(size=10))
