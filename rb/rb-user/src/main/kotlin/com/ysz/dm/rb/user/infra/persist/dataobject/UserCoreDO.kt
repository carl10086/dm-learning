package com.ysz.dm.rb.user.infra.persist.dataobject

import javax.persistence.Entity
import javax.persistence.Id

/**
 *<pre>
 * class desc here
 *</pre>
 *@author carl.yu
 *@createAt 2022/11/15
 **/
@Entity
data class UserCoreDO(
    @Id
    val id: Long,
    val comeFrom: Int,
    var nickname: String = "",
    val createAt: Long,
    var updateAt: Long,
    var username: String = "",
    var password: String = "",
    var phone: String,
    var authStatus: Int,
    /*your token issue At must >= , or is invalid */
    var tokenMinIssueAt: Long = 0L,
    /*last update at*/
    var auditAt: Long = 0L
)
