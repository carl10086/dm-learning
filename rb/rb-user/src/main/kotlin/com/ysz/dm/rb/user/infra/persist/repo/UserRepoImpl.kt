package com.ysz.dm.rb.user.infra.persist.repo

import com.ysz.dm.rb.user.domain.user.User
import com.ysz.dm.rb.user.domain.user.UserRepo
import com.ysz.dm.rb.user.infra.persist.dao.UserDao
import org.springframework.stereotype.Repository
import javax.annotation.Resource

/**
 *<pre>
 * class desc here
 *</pre>
 *@author carl.yu
 *@createAt 2022/11/15
 **/
@Repository
class UserRepoImpl(@field:Resource val userDao: UserDao) : UserRepo {


    override fun save(user: User) {
        TODO("Not yet implemented")
    }
}