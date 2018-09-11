//
// (c) 2004-2015 DS Data Systems UK Ltd, All rights reserved.
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
package com.konakartadmin.apiexamples;

import java.util.Date;

import com.konakartadmin.app.AdminCustomer;
import com.konakartadmin.app.AdminCustomerRegistration;

/**
 * This class shows how to call the KonaKart Admin API to register a new customer. Before running
 * you may have to edit BaseApiExample.java to change the username and password used to log into the
 * engine. The default values are admin@konakart.com / princess .
 */
public class RegisterCustomer extends BaseApiExample
{
    private static final String usage = "Usage: RegisterCustomer\n" + COMMON_USAGE;

    /**
     * @param args
     */
    public static void main(String[] args)
    {
        try
        {
            /*
             * Parse the command line arguments
             */
            parseArgs(args, usage, 0);
            
            /*
             * Get an instance of the Admin KonaKart engine and login. The method called can be
             * found in BaseApiExample.java
             */
            init(getEngineMode(), getStoreId(), getEngClassName(), isCustomersShared(),
                    isProductsShared(), isCategoriesShared(), null);

            /*
             * Instantiate an AdminCustomerRegistration object and set its attributes
             */
            AdminCustomerRegistration acr = new AdminCustomerRegistration();
            
            /* Compulsory attributes */
            // Gender should be set to "m", "f" (or "x" if Other Genders are enabled)
            acr.setGender("m");
            acr.setFirstName("Peter");
            acr.setLastName("Smith");
            // If you do not require the birth date, just set it to the current date. It is
            // compulsory and so needs to be set.
            acr.setBirthDate(new Date());
            acr.setEmailAddr("peter.smith@konakart.com");
            acr.setStreetAddress("23 Halden Close");
            acr.setCity("Newcastle");
            acr.setPostcode("ST5 ORT");
            acr.setTelephoneNumber("01782 639706");
            acr.setTelephoneNumber1("01782 639712");
            acr.setPassword("secret");
            acr.setDontEncryptPassword(false);
            
            /* Optional attributes */
            acr.setCompany("DS Data Systems Ltd.");
            // Country Id needs to be set with the id of a valid country from the Countries table
            acr.setCountryId(222);
            acr.setFaxNumber("01782 639707");
            // Newsletter should be set to "0" (don't receive) or "1" (receive) 
            acr.setNewsletter("1");
            acr.setSuburb("May Bank");
            acr.setState("Staff");
            
            acr.setAddrEmail("address-email@konakart.com");
            acr.setAddrTelephone("01908 0000");
            acr.setAddrTelephone1("01908 1111");
            
            /* Optional Custom fields for customer object */
            acr.setCustomerCustom1("Number Plate");
            acr.setCustomerCustom2("Driver's license");
            acr.setCustomerCustom3("Passport");
            acr.setCustomerCustom4("custom 4");
            acr.setCustomerCustom5("custom 5");
            
            /* Optional Custom fields for address object */
            acr.setAddressCustom1("custom 1");
            acr.setAddressCustom2("custom 2");
            acr.setAddressCustom3("custom 3");
            acr.setAddressCustom4("custom 4");
            acr.setAddressCustom5("custom 5");
            
            // Register the customer and get the customer Id
            int custId = eng.registerCustomer(sessionId, acr);;

            System.out.println("Id of the new customer = " + custId);

            // Read the customer from the database
            AdminCustomer cust = eng.getCustomerForId(sessionId, custId);

            if (cust != null)
            {
                System.out.println(cust.toString());
            } else
            {
                System.out.println("The customer could not be read from the DB");
            }

        } catch (Exception e)
        {
            e.printStackTrace();
        }
    }
}
