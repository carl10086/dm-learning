package com.ysz.dm.base.cache.exceptions

import com.ysz.dm.base.core.exceptions.CustomBaseException
import kotlinx.serialization.Serializable

/**
 * @author carl
 * @since 2023-01-21 9:39 PM
 **/
@Serializable
class CustomCacheException : CustomBaseException {
    constructor(message: String) : super(message)
    constructor(message: String, cause: Throwable) : super(message, cause)
    constructor(cause: Throwable) : super(cause)


    companion object {
        private const val serialVersionUID = -1037841049522100992L;

        fun of(e: Exception): CustomCacheException =
            if (e is CustomCacheException) e else CustomCacheException(e)
    }

}