package com.ysz.dm.base.core.exceptions

import kotlinx.serialization.Serializable

/**
 *<pre>
 * class desc here
 *</pre>
 *@author carl.yu
 *@since 2023/1/2
 **/
@Serializable
open class ApiParamException : CustomBaseException {
    constructor(message: String) : super(message)
    constructor(message: String, cause: Throwable) : super(message, cause)

    companion object {
        private const val serialVersionUID = -1037841049522100992L;
    }
}