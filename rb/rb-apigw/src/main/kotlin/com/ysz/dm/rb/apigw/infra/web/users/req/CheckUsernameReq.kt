package com.ysz.dm.rb.apigw.infra.web.users.req

import com.ysz.dm.rb.base.core.tools.hibernate.external.ValidUsername

/**
 *<pre>
 * class desc here
 *</pre>
 *@author carl.yu
 *@createAt 2022/11/13
 **/
data class CheckUsernameReq(
    @field:ValidUsername
    val username: String
)
