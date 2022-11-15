package com.ysz.dm.rb.user.domain.service

import com.ysz.dm.rb.base.core.ddd.DomainService
import com.ysz.dm.rb.user.domain.user.UserId
import com.ysz.et.rd.user.infra.base.core.id.Snowflake

/**
 *<pre>
 * class desc here
 *</pre>
 *@author carl.yu
 *@createAt 2022/11/15
 **/
@DomainService
class UserIdGenerator {

    fun nextId(): UserId {
        var nextId = snowflake.nextId()
        return UserId(nextId)
    }

    private companion object {
        val snowflake = Snowflake()
    }
}