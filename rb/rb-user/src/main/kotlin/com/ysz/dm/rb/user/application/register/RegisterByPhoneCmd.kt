package com.ysz.dm.rb.user.application.register

import com.ysz.dm.rb.base.core.tools.hibernate.external.ValidPhone

/**
 *<pre>
 * class desc here
 *</pre>
 *@author carl.yu
 *@createAt 2022/11/15
 **/
data class RegisterByPhoneCmd(
    @field: ValidPhone val phone: String
) {
}