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
open class CustomBaseException : RuntimeException {
    constructor(message: String) : super(message)
    constructor(message: String, cause: Throwable) : super(message, cause)
    constructor(cause: Throwable) : super(cause)


    companion object {
        private const val serialVersionUID: Long = -7413224125316564738L
    }
}