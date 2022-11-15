package com.ysz.dm.rb

import com.ysz.dm.rb.base.core.exceptions.BaseRbException

/**
 * @author carl
 * @create 2022-11-15 4:34 PM
 **/
class CustomCacheException : BaseRbException {
    constructor(message: String) : super(message)
    constructor(message: String, cause: Throwable) : super(message, cause)

    companion object {
        private const val serialVersionUID = 5414636322451170985L
    }
}