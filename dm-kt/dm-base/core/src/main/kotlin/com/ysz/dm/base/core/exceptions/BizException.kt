package com.ysz.dm.base.core.exceptions

import kotlinx.serialization.Serializable

/**
 *<pre>
 * 业务异常类
 *</pre>
 *@author carl.yu
 *@since 2023/1/2
 **/
@Serializable
open class BizException : CustomBaseException {

    constructor(message: String) : super(message)
    constructor(message: String, cause: Throwable) : super(message, cause)
    constructor(cause: Throwable) : super(cause)
}