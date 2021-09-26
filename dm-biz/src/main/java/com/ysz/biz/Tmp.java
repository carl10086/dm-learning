package com.ysz.biz;

import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.apache.commons.lang3.RegExUtils;
import org.apache.commons.lang3.StringUtils;
import org.springframework.util.PatternMatchUtils;

public class Tmp {

  public static boolean checkEmail(String email) {
    String regex = "\\w+@\\w+\\.[a-z]+(\\.[a-z]+)?";
    return Pattern.matches(regex, email);
  }


  private static boolean isValidEmailOrQQ(String email) {
    if (email.length() > 50) {
      return false;
    }
    Pattern pattern = Pattern.compile("^([1-9a-zA-Z._]+[0]?)+@[0-9a-zA-Z]+[.][a-z-]{2,10}");
    Matcher matcher = pattern.matcher(email);
    if (matcher.find()) {
      return true;
    }
    pattern = Pattern.compile("^[1-9][0-9]{4,10}$");
    matcher = pattern.matcher(email);
    if (matcher.find()) {
      return true;
    }
    return false;
  }

  public static void main(String[] args) {
    final String txt = "dnjnfslkffkjkjkslioeo9edkdjfks";
    System.out.println(checkEmail(txt));
  }
}
