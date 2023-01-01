package com.ysz.dm.base.core.validate

import com.ysz.dm.base.core.exceptions.ApiParamException

/**
 *<pre>
 * class desc here
 *</pre>
 *@author carl.yu
 *@since 2023/1/2
 **/
class ApiParamValidateTools {

    companion object {
        fun <T> notNull(src: T?, errorMsg: String): T {
            chk(src != null, errorMsg)
            return src!!
        }

        fun chk(expression: Boolean, errorMsg: String) {
            if (!expression) throw ApiParamException(errorMsg)
        }
    }
}