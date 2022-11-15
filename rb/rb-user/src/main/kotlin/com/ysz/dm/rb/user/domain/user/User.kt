package com.ysz.dm.rb.user.domain.user

import com.ysz.dm.rb.base.core.ddd.BaseEntity

/**
 *<pre>
 * class desc here
 *</pre>
 *@author carl.yu
 *@createAt 2022/11/15
 **/
class User(
    private val id: UserId,
    val comeFrom: UserRegisterComeFrom = UserRegisterComeFrom.RB_ORIGINAL_WEB,
    var auth: UserAuth,
    var nickname: String = "",
    val createAt: Long,
    var updateAt: Long
) : BaseEntity<UserId> {
    override fun id(): UserId = id
}