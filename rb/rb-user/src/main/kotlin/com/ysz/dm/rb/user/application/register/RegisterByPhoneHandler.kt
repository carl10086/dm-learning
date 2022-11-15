package com.ysz.dm.rb.user.application.register

import com.ysz.dm.rb.base.core.ddd.CmdHandler
import com.ysz.dm.rb.base.core.tools.hibernate.HibernateValidateTools
import com.ysz.dm.rb.user.domain.service.UserIdGenerator
import com.ysz.dm.rb.user.domain.user.User
import com.ysz.dm.rb.user.domain.user.UserAuth
import com.ysz.dm.rb.user.domain.user.UserAuthStatus
import com.ysz.dm.rb.user.domain.user.UserRepo
import org.slf4j.LoggerFactory
import javax.annotation.Resource

/**
 *<pre>
 * class desc here
 *</pre>
 *@author carl.yu
 *@createAt 2022/11/15
 **/
class RegisterByPhoneHandler(
    @field: Resource val userRepo: UserRepo,
    @field: Resource val userIdGenerator: UserIdGenerator
) : CmdHandler<RegisterByPhoneCmd, User> {
    override fun handle(req: RegisterByPhoneCmd): User {
        if (log.isDebugEnabled) log.debug("recv register req:{}", req)

        HibernateValidateTools.chkThenThrow(req)

        val user = init(req)

        userRepo.save(user)

        if (log.isDebugEnabled) log.debug("success reg user:{}", user)

        return user
    }

    private fun init(
        req: RegisterByPhoneCmd
    ): User {
        val now = System.currentTimeMillis()
        val user = User(
            id = userIdGenerator.nextId(),
            auth = UserAuth(
                phone = req.phone,
                auditAt = now,
                authStatus = UserAuthStatus.AUDITING
            ),
            createAt = now,
            updateAt = now
        )
        return user
    }

    companion object {
        private val log = LoggerFactory.getLogger(RegisterByPhoneHandler::class.java)
    }
}