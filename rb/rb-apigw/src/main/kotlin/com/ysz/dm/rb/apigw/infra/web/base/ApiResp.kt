package com.ysz.dm.rb.apigw.infra.web.base

/**
 *<pre>
 * class desc here
 *</pre>
 *@author carl.yu
 *@createAt 2022/11/13
 **/
data class ApiResp<T>(
    val data: T,
    var status: Int = Status.SUCCESS.code,
    var errorMsg: String? = null
) {

    enum class Status(val code: Int) {

        SUCCESS(0),
        INVALID_PARAM(1),
        SYS_ERROR(2)
    }

}
