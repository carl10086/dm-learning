package com.ysz.dm.rb.user.domain.user

/**
 *<pre>
 * class desc here
 *</pre>
 *@author carl.yu
 *@createAt 2022/11/15
 **/
enum class UserAuthStatus(val code: Int) {
    UNKNOWN(-1),
    NORMAL(0),
    LOCK(1),
    AUDITING(7),
    BLOCK(6),
    SUSPECT(10);
}