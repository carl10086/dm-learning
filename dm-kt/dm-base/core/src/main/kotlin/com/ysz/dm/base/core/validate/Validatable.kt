package com.ysz.dm.base.core.validate

import com.ysz.dm.base.core.exceptions.ApiParamException

/**
 *<pre>
 * class desc here
 *</pre>
 *@author carl.yu
 *@since 2023/1/2
 **/
interface Validatable {

    @Throws(ApiParamException::class)
    fun validate()
}