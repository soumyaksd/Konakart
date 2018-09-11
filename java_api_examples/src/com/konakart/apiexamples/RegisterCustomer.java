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

import java.util.GregorianCalendar;

import com.konakart.app.CustomerRegistration;
import com.konakart.appif.CustomerRegistrationIf;

/**
 * This class shows how to call the KonaKart API to register a new customer. Before running you may
 * have to edit BaseApiExample.java to change the username and password used to log into the engine.
 * The default values are doe@konakart.com / password .
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
            init();

            /*
             * Instantiate a CustomerRegistration object and set its attributes
             */
            CustomerRegistrationIf cr = new CustomerRegistration();

            /* Compulsory attributes */
            // Gender should be set to "m", "f" (or "x" if Other gender is enabled)
            cr.setGender("f");
            cr.setFirstName("Judy");
            cr.setLastName("Smith");
            // If you do not require the birth date, just set it to the current date. It is
            // compulsory and so needs to be set.
            cr.setBirthDate(new GregorianCalendar());
            cr.setEmailAddr("judy.smith@konakart.com");
            cr.setStreetAddress("23 Halden Close");
            cr.setCity("Newcastle");
            cr.setPostcode("ST5 ORT");
            cr.setTelephoneNumber("01782 639706");
            cr.setPassword("secret");

            /* Optional attributes */
            cr.setCompany("DS Data Systems Ltd.");
            // Country Id needs to be set with the id of a valid country from the Countries table
            cr.setCountryId(getCountryIdByName("United Kingdom"));
            cr.setFaxNumber("01782 639707");
            // Newsletter should be set to "0" (don't receive) or "1" (receive)
            cr.setNewsletter("1");
            cr.setSuburb("May Bank");
            cr.setState("Staff");

            /* Optional Custom fields for customer object */
            cr.setCustomerCustom1("Number Plate");
            cr.setCustomerCustom2("Driver's license");
            cr.setCustomerCustom3("Passport");
            cr.setCustomerCustom4("custom 4");
            cr.setCustomerCustom5("custom 5");

            /* Optional Custom fields for address object */
            cr.setAddressCustom1("custom 1");
            cr.setAddressCustom2("custom 2");
            cr.setAddressCustom3("custom 3");
            cr.setAddressCustom4("custom 4");
            cr.setAddressCustom5("custom 5");

            // Register the customer and get the customer Id
            int custId = eng.registerCustomer(cr);
            ;

            System.out.println("Id of the new customer = " + custId);

        } catch (Exception e)
        {
            e.printStackTrace();
        }
    }
}
