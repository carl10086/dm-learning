package com.ysz.dm.rb.user.domain.user

/**
 *<pre>
 * user authentication info
 *</pre>
 *@author carl.yu
 *@createAt 2022/11/15
 **/
class UserAuth(
    var username: String = "",
    var password: String = "",
    var phone: String,
    var authStatus: UserAuthStatus,
    /*your token issue At must >= , or is invalid */
    var tokenMinIssueAt: Long = 0L,
    /*last update at*/
    var auditAt: Long = 0L
) {
}