package com.ysz.dm.rb.base.core.tools.id

import com.ysz.et.rd.user.infra.base.core.id.Snowflake
import java.util.*

/**
 *<pre>
 * class desc here
 *</pre>
 *@author carl.yu
 *@createAt 2022/11/10
 **/
class IdTools {


    companion object {
        val snowflake = Snowflake()

        fun uuid(): String = UUID.randomUUID().toString().replace("-", "")
    }


}