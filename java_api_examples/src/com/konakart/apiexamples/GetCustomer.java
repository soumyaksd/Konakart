//
// (c) 2006 DS Data Systems UK Ltd, All rights reserved.
//
// DS Data Systems and KonaKart and their respective logos, are 
// trademarks of DS Data Systems UK Ltd. All rights reserved.
//
// The information in this document is free software; you can redistribute 
// it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// 
// This software is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//

package com.konakart.apiexamples;

import com.konakart.appif.CustomerIf;
import com.konakart.bl.MgrFactory;
import com.konakart.blif.CustomerMgrIf;

/**
 * An example of how to retrieve a Customer from an email address using a public method on the
 * CustomerMgr
 */
public class GetCustomer extends BaseApiExample
{
    private static final String usage = "Usage: GetCustomer\n" + COMMON_USAGE;

    /**
     * @param args
     */
    public static void main(String[] args)
    {
        parseArgs(args, usage, 0);

        try
        {
            /*
             * Get an instance of the KonaKart engine and login. The method called can be found in
             * BaseApiExample.java
             */
            init();

            MgrFactory mgrFactory = new MgrFactory(eng);
            CustomerMgrIf mgr = mgrFactory.getCustMgr(/* new */true);
            CustomerIf cust = mgr.getCustomerForEmail(getDEFAULT_USERNAME());

            if (cust == null)
            {
                System.out.println("Customer not found");
            } else
            {
                System.out.println(cust.toString());
            }

            System.out.println("Completed Successfully");

        } catch (Exception e)
        {
            e.printStackTrace();
        }
    }
}
