package com.ysz.dm.rb.base.core.exceptions

/**
 *<pre>
 * class desc here
 *</pre>
 *@author carl.yu
 *@createAt 2022/11/13
 **/
open class BaseRbException : RuntimeException {

    constructor(message: String) : super(message)

    constructor(message: String, cause: Throwable) : super(message, cause)


    companion object {
        private const val serialVersionUID = 5414636322451170985L
    }

}