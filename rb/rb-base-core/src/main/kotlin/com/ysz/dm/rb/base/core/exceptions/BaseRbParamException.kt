package com.ysz.dm.rb.base.core.exceptions

/**
 *<pre>
 * class desc here
 *</pre>
 *@author carl.yu
 *@createAt 2022/11/13
 **/
open class BaseRbParamException : BaseRbException {

    constructor(message: String) : super(message)
    constructor(message: String, cause: Throwable) : super(message, cause)


}